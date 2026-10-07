package org.localTranslation.tools;

import java.net.URL;
import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.bson.Document;
import org.bson.conversions.Bson;
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
import com.mongodb.MongoNamespace;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Aggregates;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.IndexOptions;
import com.mongodb.client.model.RenameCollectionOptions;
import com.mongodb.client.model.Updates;
import io.vavr.control.Either;

public class CallLongestAndAiNlp {

    private record CollectionNamePair(String collection, String workingCollection) {
    }

    private record RequestConfig(Set<LanguageCode> alternatives, boolean verify, TransliterationMethod method) {
    }

    private record EntityConfig(RequestConfig name, RequestConfig address) {
        boolean isEmpty() {
            return name == null && address == null;
        }
    }

    private record EntryFields(String textKey, String languageKey, boolean missingLanguageIsEnglish) {
    }

    private record NlpJob(TransliterationClient client, AiProvider provider, String database, String collectionName,
                          EntityConfig config, MongoCollection<Document> target, AtomicInteger updated,
                          AtomicInteger failed) {
    }

    private static final EntryFields NAME = new EntryFields("original", "language", true);
    private static final EntryFields ADDRESS = new EntryFields("address", "ln", false);

    private static final String LONGEST_NAME_URL = "http://localhost:8482";
    private static final String NLP_URL = "http://192.168.1.16:30646";

    static void main() throws Exception {
        boolean runLongestName = true;
        boolean runNlp = true;
        boolean promoteNlpCollections = true;

        String mongoUri = "mongodb://localhost:27017";
        String database = "entity_test1_registry";

        AiProvider longestNameProvider = AiProvider.CHATGPT;


        AiProvider nlpProvider = AiProvider.CHATGPT;
        RequestConfig aiNlp = new RequestConfig(Set.of(LanguageCode.ENGLISH), false, TransliterationMethod.AI_NLP);
        EntityConfig organization = new EntityConfig(aiNlp, aiNlp); // name, address
        EntityConfig person = new EntityConfig(aiNlp, aiNlp);       // name, address

        try (MongoClient mongoClient = MongoClients.create(mongoUri)) {
            MongoDatabase mongoDatabase = mongoClient.getDatabase(database);

            List<String> nodeCollections = findNodeCollections(mongoDatabase);
            System.out.println("[db] " + database + ": node collections to translate: " + nodeCollections);
            if (nodeCollections.isEmpty()) {
                System.out.println("[db] no node_person*/node_organization* collection found - nothing to do");
                return;
            }

            if (runLongestName) {
                runLongestNameStep(mongoDatabase, database, nodeCollections, longestNameProvider);
            }
            if (runNlp) {
                runNlpStep(mongoDatabase, nodeCollections, database, nlpProvider, organization, person,
                        promoteNlpCollections);
            }
        }
    }

    private static List<String> findNodeCollections(MongoDatabase mongoDatabase) {
        List<String> found = new ArrayList<>();
        for (String name : mongoDatabase.listCollectionNames()) {
            String lower = name.toLowerCase(Locale.ROOT);
            boolean node = lower.startsWith("node_") && (lower.contains("person") || lower.contains("organization"));
            boolean temporary = lower.endsWith("_translation") || lower.endsWith("_nlp");
            if (node && !temporary) {
                found.add(name);
            }
        }
        Collections.sort(found);
        return found;
    }

    private static String workingCollectionName(String collection) {
        return withoutNewSuffix(collection) + "_translation";
    }

    private static String nlpCollectionName(String collection) {
        return withoutNewSuffix(collection) + "_nlp";
    }

    private static String withoutNewSuffix(String collection) {
        return collection.endsWith("_new") ? collection.substring(0, collection.length() - "_new".length()) : collection;
    }

    private static String resolvePublisher(MongoCollection<Document> collection) {
        List<String> distinct = collection.distinct("metadata.publisher", String.class).into(new ArrayList<>());
        if (distinct.isEmpty()) {
            return null;
        }
        if (distinct.size() > 1) {
            System.out.println("[publisher] WARNING: collection has multiple publishers " + distinct
                    + ", using first: " + distinct.getFirst());
        }
        return distinct.getFirst();
    }

    private static String resolvePublisher(Document doc) {
        Document metadata = doc.get("metadata", Document.class);
        if (metadata == null) {
            return null;
        }
        List<String> publishers = metadata.getList("publisher", String.class, List.of());
        return publishers.isEmpty() ? null : publishers.getFirst();
    }

    private static void unsetEn(MongoCollection<Document> collection, String array) {
        var result = collection.updateMany(Filters.exists(array + ".0"), Updates.unset(array + ".$[].en"));
        System.out.println("[unset] " + collection.getNamespace().getCollectionName() + "." + array + ": matched="
                + result.getMatchedCount() + " modified=" + result.getModifiedCount());
    }

    private static void runLongestNameStep(MongoDatabase mongoDatabase, String database, List<String> collections,
                                            AiProvider provider) {
        LongestNameAiTranslationClient client = new LongestNameAiTranslationClientImpl(
                HttpClient.newHttpClient(), LONGEST_NAME_URL, new ObjectMapper());

        List<CollectionNamePair> collectionNamePairs = collections.stream()
                .map(collection -> new CollectionNamePair(collection, workingCollectionName(collection)))
                .toList();

        for (CollectionNamePair pair : collectionNamePairs) {
            String publisher = resolvePublisher(mongoDatabase.getCollection(pair.collection()));
            client.process(pair.collection(), pair.workingCollection(), database, provider, publisher);
            System.out.println("[longest-name] Submitted: " + pair.collection() + " -> " + pair.workingCollection()
                    + " (publisher=" + publisher + ")");
        }
    }

    private static void runNlpStep(MongoDatabase mongoDatabase, List<String> collections, String database,
                                    AiProvider provider, EntityConfig organization, EntityConfig person,
                                    boolean promoteNlpCollections) throws Exception {
        System.out.println("[nlp] provider=" + provider + " | organization: " + describe(organization)
                + " | person: " + describe(person));

        URL nlpBaseUrl = new URL(NLP_URL);

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

        int concurrency = 16;

        AtomicInteger updated = new AtomicInteger();
        AtomicInteger done = new AtomicInteger();
        Map<String, AtomicInteger> failedByCollection = new LinkedHashMap<>();
        ExecutorService executor = Executors.newFixedThreadPool(concurrency);
        Semaphore queued = new Semaphore(concurrency * 4);

        for (String collectionName : collections) {
            EntityConfig config = collectionName.toLowerCase(Locale.ROOT).contains("organization") ? organization : person;
            if (config.isEmpty()) {
                System.out.println("[nlp] " + collectionName + ": nothing selected for this entity type, skipped");
                continue;
            }

            String nlpName = nlpCollectionName(collectionName);
            MongoCollection<Document> collection = mongoDatabase.getCollection(nlpName);
            collection.drop();
            mongoDatabase.getCollection(collectionName).aggregate(List.of(Aggregates.out(nlpName))).toCollection();
            System.out.println("[nlp] " + collectionName + " -> " + nlpName + ": copied " + collection.countDocuments()
                    + " documents");
            if (config.name() != null) {
                unsetEn(collection, "names");
                unsetEn(collection, "nameRecords");
            }
            if (config.address() != null) {
                unsetEn(collection, "address");
                unsetEn(collection, "addressRecords");
            }

            AtomicInteger failed = new AtomicInteger();
            failedByCollection.put(collectionName, failed);
            NlpJob job = new NlpJob(client, provider, database, collectionName, config, collection, updated, failed);

            for (Document doc : collection.find().noCursorTimeout(true).batchSize(500)) {
                submit(executor, queued, failed, done, () -> translateDocument(job, doc));
            }
        }

        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

        int totalFailed = 0;
        for (Map.Entry<String, AtomicInteger> entry : failedByCollection.entrySet()) {
            String collectionName = entry.getKey();
            String nlpName = nlpCollectionName(collectionName);
            int failed = entry.getValue().get();
            totalFailed += failed;
            long liveCount = mongoDatabase.getCollection(collectionName).countDocuments();
            long nlpCount = mongoDatabase.getCollection(nlpName).countDocuments();
            if (!promoteNlpCollections) {
                System.out.println("[nlp] " + nlpName + " is left as it is: " + nlpCount + " documents, " + failed
                        + " failed requests");
            } else if (failed > 0 || liveCount != nlpCount) {
                System.out.println("[nlp] " + nlpName + " is NOT renamed to " + collectionName + ": " + failed
                        + " failed requests, " + nlpCount + " documents against " + liveCount);
            } else {
                copyIndexes(mongoDatabase.getCollection(collectionName), mongoDatabase.getCollection(nlpName));
                mongoDatabase.getCollection(nlpName).renameCollection(
                        new MongoNamespace(mongoDatabase.getName(), collectionName),
                        new RenameCollectionOptions().dropTarget(true));
                System.out.println("[nlp] renamed " + nlpName + " to " + collectionName);
            }
        }

        System.out.println("[nlp] Updated " + updated.get() + " fields, " + totalFailed + " failed");
    }

    private static void translateDocument(NlpJob job, Document doc) {
        String publisher = resolvePublisher(doc);
        List<Bson> changes = new ArrayList<>();

        RequestConfig nameRequest = job.config().name();
        if (nameRequest != null) {
            List<Document> names = entries(doc, "names");
            List<Document> nameRecords = entries(doc, "nameRecords");
            translateEntries(job, publisher, nameRequest, names, NAME);
            translateRecords(job, publisher, nameRequest, nameRecords, names, NAME);
            addChange(changes, "names", names);
            addChange(changes, "nameRecords", nameRecords);
        }

        RequestConfig addressRequest = job.config().address();
        if (addressRequest != null) {
            List<Document> addresses = entries(doc, "address");
            List<Document> addressRecords = entries(doc, "addressRecords");
            translateEntries(job, publisher, addressRequest, addresses, ADDRESS);
            translateRecords(job, publisher, addressRequest, addressRecords, addresses, ADDRESS);
            addChange(changes, "address", addresses);
            addChange(changes, "addressRecords", addressRecords);
        }

        if (!changes.isEmpty()) {
            job.target().updateOne(Filters.eq("_id", doc.get("_id")), Updates.combine(changes));
        }
    }

    private static List<Document> entries(Document doc, String array) {
        return doc.getList(array, Document.class, List.of());
    }

    private static void addChange(List<Bson> changes, String array, List<Document> entries) {
        if (!entries.isEmpty()) {
            changes.add(Updates.set(array, entries));
        }
    }

    private static void translateEntries(NlpJob job, String publisher, RequestConfig request, List<Document> entries,
                                         EntryFields fields) {
        List<Document> toSend = new ArrayList<>();
        List<TransliterationRequest> requests = new ArrayList<>();
        for (Document entry : entries) {
            String text = entry.getString(fields.textKey());
            LanguageCode from = sourceLanguage(entry, fields);
            if (text == null || text.isBlank() || from == null) {
                continue;
            }
            toSend.add(entry);
            requests.add(newRequest(job, publisher, request, text, from));
        }
        send(job, requests, toSend, fields);
    }

    private static void translateRecords(NlpJob job, String publisher, RequestConfig request, List<Document> records,
                                         List<Document> entries, EntryFields fields) {
        Map<String, Document> entryByText = new HashMap<>();
        for (Document entry : entries) {
            String text = entry.getString(fields.textKey());
            if (text != null) {
                entryByText.putIfAbsent(text, entry);
            }
        }

        List<Document> toSend = new ArrayList<>();
        List<TransliterationRequest> requests = new ArrayList<>();
        for (Document record : records) {
            String text = record.getString(fields.textKey());
            LanguageCode from = sourceLanguage(record, fields);
            if (text == null || from == null) {
                continue;
            }
            Document sameEntry = entryByText.get(text);
            if (sameEntry != null) {
                copyAnswer(sameEntry, record);
            } else if (!text.isBlank()) {
                toSend.add(record);
                requests.add(newRequest(job, publisher, request, text, from));
            }
        }
        send(job, requests, toSend, fields);
    }

    private static LanguageCode sourceLanguage(Document entry, EntryFields fields) {
        String language = entry.getString(fields.languageKey());
        if (language == null || language.isBlank()) {
            return fields.missingLanguageIsEnglish() ? LanguageCode.ENGLISH : null;
        }
        LanguageCode code = LanguageCode.of(language);
        return LanguageCode.UNAVAILABLE.equals(code) ? null : code;
    }

    private static TransliterationRequest newRequest(NlpJob job, String publisher, RequestConfig request, String text,
                                                      LanguageCode from) {
        return new TransliterationRequest(text, from, LanguageCode.ENGLISH, request.alternatives(), request.verify(),
                request.method(), job.provider(), job.database(), publisher, job.collectionName());
    }

    private static void copyAnswer(Document from, Document to) {
        String en = from.getString("en");
        if (!Objects.equals(to.getString("en"), en)) {
            if (en == null) {
                to.remove("en");
            } else {
                to.put("en", en);
            }
        }
    }

    private static void send(NlpJob job, List<TransliterationRequest> requests, List<Document> entries,
                             EntryFields fields) {
        if (requests.isEmpty()) {
            return;
        }
        Either<?, List<TransliterationResponse>> result = job.client().transliterate(requests);
        if (result.isLeft()) {
            countFailure(job.failed(), job.collectionName() + ": " + result.getLeft());
            return;
        }
        List<TransliterationResponse> responses = result.get();
        if (responses.size() != entries.size()) {
            countFailure(job.failed(), job.collectionName() + ": " + entries.size() + " requests but "
                    + responses.size() + " answers");
            return;
        }
        for (int i = 0; i < entries.size(); i++) {
            Document entry = entries.get(i);
            TransliterationResponse response = responses.get(i);
            entry.put("en", response.getText());
            response.getCodeOpt().ifPresent(code -> entry.put(fields.languageKey(), code.get()));
        }
        job.updated().addAndGet(entries.size());
    }

    private static void countFailure(AtomicInteger failed, String message) {
        if (failed.incrementAndGet() <= 5) {
            System.out.println("[nlp] request failed: " + message);
        }
    }

    private static String describe(EntityConfig config) {
        return "name=" + describe(config.name()) + ", address=" + describe(config.address());
    }

    private static String describe(RequestConfig request) {
        return request == null ? "off" : request.method() + " alternatives="
                + request.alternatives().stream().map(LanguageCode::get).toList() + " verify=" + request.verify();
    }

    private static void copyIndexes(MongoCollection<Document> from, MongoCollection<Document> to) {
        for (Document index : from.listIndexes()) {
            String indexName = index.getString("name");
            if ("_id_".equals(indexName)) {
                continue;
            }
            try {
                IndexOptions options = new IndexOptions().name(indexName);
                if (index.getBoolean("unique", false)) {
                    options.unique(true);
                }
                if (index.getBoolean("sparse", false)) {
                    options.sparse(true);
                }
                Document partialFilter = index.get("partialFilterExpression", Document.class);
                if (partialFilter != null) {
                    options.partialFilterExpression(partialFilter);
                }
                Number expireAfterSeconds = index.get("expireAfterSeconds", Number.class);
                if (expireAfterSeconds != null) {
                    options.expireAfter(expireAfterSeconds.longValue(), TimeUnit.SECONDS);
                }
                to.createIndex(index.get("key", Document.class), options);
            } catch (RuntimeException e) {
                System.out.println("[nlp] WARNING: index " + indexName + " of " + from.getNamespace().getCollectionName()
                        + " could not be copied, create it again by hand: " + e.getMessage());
            }
        }
    }

    private static void submit(ExecutorService executor, Semaphore queued, AtomicInteger failed, AtomicInteger done,
                               Runnable work) {
        queued.acquireUninterruptibly();
        executor.execute(() -> {
            try {
                work.run();
            } catch (RuntimeException e) {
                countFailure(failed, e.toString());
            } finally {
                queued.release();
                int finished = done.incrementAndGet();
                if (finished % 500 == 0) {
                    System.out.println("[nlp] " + finished + " documents done");
                }
            }
        });
    }
}
