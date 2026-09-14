package org.localTranslation.tools;

import java.net.URL;
import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import org.bson.Document;
import org.zalando.problem.jackson.ProblemModule;

import com.ardis.batch.support.client.LongestNameAiTranslationClient;
import com.ardis.batch.support.client.LongestNameAiTranslationClientImpl;
import com.ardis.entity.api.LanguageCode;
import com.ardis.nlp.client.TransliterationClient;
import com.ardis.nlp.data.action.method.AiProvider;
import com.ardis.nlp.data.action.method.TransliterationMethod;
import com.ardis.nlp.data.transliteration.TransliterationRequest;
import com.ardis.nlp.data.transliteration.TransliterationResponse;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.fasterxml.jackson.databind.module.SimpleModule;
import com.fasterxml.jackson.databind.ser.std.StdSerializer;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.UpdateOptions;
import com.mongodb.client.model.Updates;
import io.vavr.control.Either;

public class CallLongestAndAiNlp {

    private record CollectionNamePair(String collection, String workingCollection) {
    }

    static void main() throws Exception {
        boolean runLongestName = true;
        boolean runNlp = true;

        String mongoUri = "mongodb://localhost:27017";
        String database = "entity_test7_registry";

        AiProvider longestNameProvider = AiProvider.GEMINI;
        AiProvider nlpProvider = AiProvider.GEMINI;

        List<String> nlpCollections = List.of("node_person_new", "node_organization_new");

        try (MongoClient mongoClient = MongoClients.create(mongoUri)) {
            MongoDatabase mongoDatabase = mongoClient.getDatabase(database);

            if (runLongestName) {
                runLongestNameStep(mongoDatabase, database, longestNameProvider);
            }
            if (runNlp) {
                unsetNamesEn(mongoDatabase, nlpCollections);
                runNlpStep(mongoDatabase, nlpCollections, database, nlpProvider);
            }
        }
    }

    private static String resolvePublisher(MongoCollection<Document> collection) {
        List<String> distinct = collection.distinct("metadata.publisher", String.class).into(new ArrayList<>());
        if (distinct.isEmpty()) {
            return null;
        }
        if (distinct.size() > 1) {
            System.out.println("[publisher] WARNING: collection has multiple publishers " + distinct
                    + ", using first: " + distinct.get(0));
        }
        return distinct.get(0);
    }

    private static String resolvePublisher(Document doc) {
        Document metadata = doc.get("metadata", Document.class);
        if (metadata == null) {
            return null;
        }
        List<String> publishers = metadata.getList("publisher", String.class, List.of());
        return publishers.isEmpty() ? null : publishers.get(0);
    }

    private static void unsetNamesEn(MongoDatabase mongoDatabase, List<String> collections) {
        for (String collectionName : collections) {
            MongoCollection<Document> collection = mongoDatabase.getCollection(collectionName);
            var result = collection.updateMany(new Document(), Updates.unset("names.$[].en"));
            System.out.println("[unset] " + collectionName + ": matched=" + result.getMatchedCount()
                    + " modified=" + result.getModifiedCount());
        }
    }

    private static void runLongestNameStep(MongoDatabase mongoDatabase, String database, AiProvider provider) {
        String baseUrl = "http://localhost:8080";

        LongestNameAiTranslationClient client = new LongestNameAiTranslationClientImpl(
                HttpClient.newHttpClient(), baseUrl, new ObjectMapper());

        List<CollectionNamePair> collectionNamePairs = List.of(
                new CollectionNamePair("node_person_new", "node_person_translation"),
                new CollectionNamePair("node_organization_new", "node_organization_translation")
        );

        for (CollectionNamePair pair : collectionNamePairs) {
            String publisher = resolvePublisher(mongoDatabase.getCollection(pair.collection()));
            client.process(pair.collection(), pair.workingCollection(), database, provider, publisher);
            System.out.println("[longest-name] Submitted: " + pair.collection() + " -> " + pair.workingCollection()
                    + " (publisher=" + publisher + ")");
        }
    }

    private static void runNlpStep(MongoDatabase mongoDatabase, List<String> collections, String database,
                                    AiProvider provider) throws Exception {
        URL nlpBaseUrl = new URL("http://localhost:8189");

        SimpleModule languageCodeModule = new SimpleModule()
                .addSerializer(LanguageCode.class, new StdSerializer<LanguageCode>(LanguageCode.class) {
                    @Override
                    public void serialize(LanguageCode value, JsonGenerator gen, SerializerProvider provider) throws java.io.IOException {
                        gen.writeString(value.get());
                    }
                })
                .addDeserializer(LanguageCode.class, new StdDeserializer<LanguageCode>(LanguageCode.class) {
                    @Override
                    public LanguageCode deserialize(JsonParser p, DeserializationContext ctxt) throws java.io.IOException {
                        return LanguageCode.of(p.getValueAsString());
                    }
                });

        ObjectMapper nlpObjectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .registerModule(new SimpleModule("com.fasterxml.jackson.datatype.jsr310.JavaTimeModule"))
                .registerModule(new ProblemModule())
                .registerModule(languageCodeModule);
        TransliterationClient client = TransliterationClient.of(nlpObjectMapper, nlpBaseUrl, HttpClient.newHttpClient());

        Set<LanguageCode> alternatives = Set.of(LanguageCode.ENGLISH);
        boolean verify = false;
        TransliterationMethod method = TransliterationMethod.AI_NLP;
        int concurrency = 16;

        AtomicInteger updated = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        List<Callable<Void>> tasks = new ArrayList<>();

        for (String collectionName : collections) {
            MongoCollection<Document> collection = mongoDatabase.getCollection(collectionName);
            for (Document doc : collection.find()) {
                Object id = doc.get("_id");
                String publisher = resolvePublisher(doc);

                for (Document name : doc.getList("names", Document.class, List.of())) {
                    String original = name.getString("original");
                    String language = name.getString("language");
                    if (original == null || original.isBlank()) {
                        continue;
                    }
                    // matches NodeEntityTransliterator.fromLanguage(...): null language defaults to ENGLISH, not skipped
                    LanguageCode from = language == null ? LanguageCode.ENGLISH : LanguageCode.of(language);
                    // matches NodeEntityTransliterator.isUnavailable(...): skip only when language detection failed,
                    // NOT when the language is already English - production transliterates English too.
                    if (LanguageCode.UNAVAILABLE.equals(from)) {
                        continue;
                    }
                    TransliterationRequest request = new TransliterationRequest(original, from,
                            LanguageCode.ENGLISH, alternatives, verify, method, provider, database, publisher, collectionName);
                    tasks.add(() -> {
                        // single-request call: returns exactly one response, unambiguously tied to this request
                        // (the bulk call can return more responses than requests, with no id to correlate them back)
                        Either<?, TransliterationResponse> result = client.transliterate(request);
                        if (result.isRight()) {
                            collection.updateOne(Filters.eq("_id", id),
                                    Updates.set("names.$[elem].en", result.get().getText()),
                                    new UpdateOptions().arrayFilters(List.of(Filters.eq("elem.original", original))));
                            updated.incrementAndGet();
                        } else {
                            failed.incrementAndGet();
                        }
                        return null;
                    });
                }

                for (Document address : doc.getList("address", Document.class, List.of())) {
                    String original = address.getString("address");
                    String language = address.getString("ln");
                    if (original == null || original.isBlank() || language == null
                            || LanguageCode.UNAVAILABLE.equals(LanguageCode.of(language))) {
                        continue;
                    }
                    TransliterationRequest request = new TransliterationRequest(original, LanguageCode.of(language),
                            LanguageCode.ENGLISH, alternatives, verify, method, provider, database, publisher, collectionName);
                    tasks.add(() -> {
                        Either<?, TransliterationResponse> result = client.transliterate(request);
                        if (result.isRight()) {
                            collection.updateOne(Filters.eq("_id", id),
                                    Updates.set("address.$[elem].en", result.get().getText()),
                                    new UpdateOptions().arrayFilters(List.of(Filters.eq("elem.address", original))));
                            updated.incrementAndGet();
                        } else {
                            failed.incrementAndGet();
                        }
                        return null;
                    });
                }
            }
        }

        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        try {
            executor.invokeAll(tasks);
        } finally {
            executor.shutdown();
        }

        System.out.println("[nlp] Updated " + updated.get() + " fields, " + failed.get() + " failed");
    }
}
