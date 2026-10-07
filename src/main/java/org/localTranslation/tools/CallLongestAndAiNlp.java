package org.localTranslation.tools;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.http.HttpClient;
import java.util.ArrayList;
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
import java.util.function.Function;
import java.util.stream.Collectors;

import org.bson.Document;
import org.bson.conversions.Bson;
import org.zalando.problem.jackson.ProblemModule;

import com.ardis.batch.support.client.LongestNameAiTranslationClient;
import com.ardis.batch.support.client.LongestNameAiTranslationClientImpl;
import com.ardis.entity.api.LanguageCode;
import com.ardis.entity.jackson.datatype.EntityApiModule;
import com.ardis.nlp.client.TransliterationClient;
import com.ardis.nlp.data.action.method.AiProvider;
import com.ardis.nlp.data.action.method.TransliterationMethod;
import com.ardis.nlp.data.transliteration.TransliterationRequest;
import com.ardis.nlp.data.transliteration.TransliterationResponse;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
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

    static void main() throws Exception {
        boolean runLongestName = true;
        boolean runNlp = true;
        boolean promoteNlpCollections = true;

        String mongoUri = "mongodb://localhost:27017";
        String database = "entity_test1_registry";

        AiProvider longestNameProvider = AiProvider.GOOGLE_TRANSLATE;
        boolean longestOrganization = true;
        boolean longestPerson = false;

        AiProvider nlpProvider = AiProvider.GOOGLE_TRANSLATE;

        RequestConfig aiNlp = new RequestConfig(Set.of(LanguageCode.ENGLISH), false, TransliterationMethod.AI_NLP);
        RequestConfig rules = new RequestConfig(Set.of(LanguageCode.ENGLISH), false, TransliterationMethod.RULES);
        RequestConfig defaultMethod = new RequestConfig(Set.of(LanguageCode.ENGLISH), false, TransliterationMethod.DEFAULT);

        EntityConfig organization = new EntityConfig(aiNlp, rules);
        EntityConfig person = new EntityConfig(rules, rules);

        try (MongoClient mongoClient = MongoClients.create(mongoUri)) {
            MongoDatabase mongoDatabase = mongoClient.getDatabase(database);

            List<String> nodeCollections = findNodeCollections(mongoDatabase);
            System.out.println("[db] " + database + ": node collections to translate: " + nodeCollections);
            if (nodeCollections.isEmpty()) {
                System.out.println("[db] no node_person*/node_organization* collection found - nothing to do");
                return;
            }

            if (runLongestName) {
                runLongestNameStep(mongoDatabase, database, nodeCollections, longestNameProvider, longestOrganization,
                        longestPerson);
            }
            if (runNlp) {
                runNlpStep(mongoDatabase, nodeCollections, database, nlpProvider, organization, person,
                        promoteNlpCollections);
            }
        }
    }

    private static final String LONGEST_NAME_URL = "http://localhost:8482";
    private static final String NLP_URL = "http://192.168.1.16:30646";
    private static final String JAVA_TIME_MODULE_ID = "com.fasterxml.jackson.datatype.jsr310.JavaTimeModule";
    private static final int CONCURRENCY = 16;

    private static final Part NAME = new Part("names", "nameRecords", "original", "language", true);
    private static final Part ADDRESS = new Part("address", "addressRecords", "address", "ln", false);
    private static final List<Part> PARTS = List.of(NAME, ADDRESS);

    private record Part(String entriesKey, String recordsKey, String textKey, String languageKey,
                        boolean missingLanguageIsEnglish) {
    }

    private record RequestConfig(Set<LanguageCode> alternatives, boolean verify, TransliterationMethod method) {
    }

    private record EntityConfig(RequestConfig name, RequestConfig address) {

        boolean isEmpty() {
            return name == null && address == null;
        }

        RequestConfig requestFor(Part part) {
            return part == NAME ? name : address;
        }
    }

    private record NlpJob(TransliterationClient client, AiProvider provider, String database, String collectionName,
                          EntityConfig config, MongoCollection<Document> target, AtomicInteger updated,
                          AtomicInteger failed) {

        void run(Document doc) {
            try {
                translate(doc);
            } catch (RuntimeException e) {
                countFailure(e.toString());
            }
        }

        private void translate(Document doc) {
            String publisher = resolvePublisher(doc);
            Map<String, List<Document>> arrays = new LinkedHashMap<>();
            for (Part part : PARTS) {
                RequestConfig request = config.requestFor(part);
                if (request == null) {
                    continue;
                }
                List<Document> entries = arrayOf(doc, part.entriesKey());
                List<Document> records = arrayOf(doc, part.recordsKey());
                send(publisher, request, entries.stream().filter(entry -> isSendable(entry, part)).toList(), part);
                translateRecords(publisher, request, records, entries, part);
                arrays.put(part.entriesKey(), entries);
                arrays.put(part.recordsKey(), records);
            }
            List<Bson> updates = arrays.entrySet().stream()
                    .filter(array -> !array.getValue().isEmpty())
                    .map(array -> Updates.set(array.getKey(), array.getValue()))
                    .toList();
            if (!updates.isEmpty()) {
                target.updateOne(Filters.eq("_id", doc.get("_id")), Updates.combine(updates));
            }
        }

        private void translateRecords(String publisher, RequestConfig request, List<Document> records,
                                      List<Document> entries, Part part) {
            Map<String, Document> entryByText = entries.stream()
                    .filter(entry -> entry.getString(part.textKey()) != null)
                    .collect(Collectors.toMap(entry -> entry.getString(part.textKey()), Function.identity(),
                            (first, second) -> first));
            List<Document> toSend = new ArrayList<>();
            for (Document record : records) {
                String text = record.getString(part.textKey());
                if (text == null || sourceLanguage(record, part) == null) {
                    continue;
                }
                Document sameEntry = entryByText.get(text);
                if (sameEntry != null) {
                    copyAnswer(sameEntry, record);
                } else if (!text.isBlank()) {
                    toSend.add(record);
                }
            }
            send(publisher, request, toSend, part);
        }

        private void send(String publisher, RequestConfig request, List<Document> entries, Part part) {
            if (entries.isEmpty()) {
                return;
            }
            List<TransliterationRequest> requests = entries.stream()
                    .map(entry -> new TransliterationRequest(entry.getString(part.textKey()),
                            sourceLanguage(entry, part), LanguageCode.ENGLISH, request.alternatives(),
                            request.verify(), request.method(), provider, database, publisher, collectionName))
                    .toList();
            Either<?, List<TransliterationResponse>> result = client.transliterate(requests);
            if (result.isLeft()) {
                countFailure(collectionName + ": " + result.getLeft());
                return;
            }
            List<TransliterationResponse> responses = result.get();
            if (responses.size() != entries.size()) {
                countFailure(collectionName + ": " + entries.size() + " requests but " + responses.size()
                        + " answers");
                return;
            }
            for (int i = 0; i < entries.size(); i++) {
                Document entry = entries.get(i);
                TransliterationResponse response = responses.get(i);
                entry.put("en", response.getText());
                response.getCodeOpt().ifPresent(code -> entry.put(part.languageKey(), code.get()));
            }
            updated.addAndGet(entries.size());
        }

        private void countFailure(String message) {
            if (failed.incrementAndGet() <= 5) {
                System.out.println("[nlp] request failed: " + message);
            }
        }
    }

    private static List<String> findNodeCollections(MongoDatabase mongoDatabase) {
        return mongoDatabase.listCollectionNames().into(new ArrayList<String>()).stream()
                .filter(CallLongestAndAiNlp::isNodeCollection)
                .sorted()
                .toList();
    }

    private static boolean isNodeCollection(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        boolean node = lower.startsWith("node_") && (lower.contains("person") || lower.contains("organization"));
        return node && !lower.endsWith("_translation") && !lower.endsWith("_nlp");
    }

    private static boolean isOrganization(String collection) {
        return collection.toLowerCase(Locale.ROOT).contains("organization");
    }

    private static String siblingName(String collection, String suffix) {
        return collection.replaceFirst("_new$", "") + suffix;
    }

    private static String resolvePublisher(MongoCollection<Document> collection) {
        List<String> publishers = collection.distinct("metadata.publisher", String.class).into(new ArrayList<>());
        if (publishers.size() > 1) {
            System.out.println("[publisher] WARNING: collection has multiple publishers " + publishers
                    + ", using first: " + publishers.getFirst());
        }
        return publishers.isEmpty() ? null : publishers.getFirst();
    }

    private static String resolvePublisher(Document doc) {
        Document metadata = doc.get("metadata", Document.class);
        List<String> publishers = metadata == null ? List.of() : metadata.getList("publisher", String.class, List.of());
        return publishers.isEmpty() ? null : publishers.getFirst();
    }

    private static List<Document> arrayOf(Document doc, String array) {
        return doc.getList(array, Document.class, List.of());
    }

    private static LanguageCode sourceLanguage(Document entry, Part part) {
        String language = entry.getString(part.languageKey());
        if (language == null || language.isBlank()) {
            return part.missingLanguageIsEnglish() ? LanguageCode.ENGLISH : null;
        }
        LanguageCode code = LanguageCode.of(language);
        return LanguageCode.UNAVAILABLE.equals(code) ? null : code;
    }

    private static boolean isSendable(Document entry, Part part) {
        String text = entry.getString(part.textKey());
        return text != null && !text.isBlank() && sourceLanguage(entry, part) != null;
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

    private static void runLongestNameStep(MongoDatabase mongoDatabase, String database, List<String> collections,
                                           AiProvider provider, boolean longestOrganization, boolean longestPerson) {
        System.out.println("[longest-name] provider=" + provider + " | organization: " + onOff(longestOrganization)
                + " | person: " + onOff(longestPerson));
        LongestNameAiTranslationClient client = new LongestNameAiTranslationClientImpl(
                HttpClient.newHttpClient(), LONGEST_NAME_URL, new ObjectMapper());

        for (String collection : collections) {
            if (!(isOrganization(collection) ? longestOrganization : longestPerson)) {
                System.out.println("[longest-name] " + collection + ": not selected, skipped");
                continue;
            }
            String workingCollection = siblingName(collection, "_translation");
            String publisher = resolvePublisher(mongoDatabase.getCollection(collection));
            client.process(collection, workingCollection, database, provider, publisher);
            System.out.println("[longest-name] Submitted: " + collection + " -> " + workingCollection
                    + " (publisher=" + publisher + ")");
        }
    }

    private static void runNlpStep(MongoDatabase mongoDatabase, List<String> collections, String database,
                                   AiProvider provider, EntityConfig organization, EntityConfig person,
                                   boolean promoteNlpCollections) throws Exception {
        System.out.println("[nlp] provider=" + provider + " | organization: " + describe(organization)
                + " | person: " + describe(person));

        TransliterationClient client = nlpClient();
        AtomicInteger updated = new AtomicInteger();
        AtomicInteger done = new AtomicInteger();
        Map<String, AtomicInteger> failedByCollection = new LinkedHashMap<>();
        ExecutorService executor = Executors.newFixedThreadPool(CONCURRENCY);
        Semaphore queued = new Semaphore(CONCURRENCY * 4);

        for (String collectionName : collections) {
            EntityConfig config = isOrganization(collectionName) ? organization : person;
            if (config.isEmpty()) {
                System.out.println("[nlp] " + collectionName + ": nothing selected for this entity type, skipped");
                continue;
            }
            MongoCollection<Document> copy = copyForNlp(mongoDatabase, collectionName, config);
            NlpJob job = new NlpJob(client, provider, database, collectionName, config, copy, updated,
                    new AtomicInteger());
            failedByCollection.put(collectionName, job.failed());
            for (Document doc : copy.find().noCursorTimeout(true).batchSize(500)) {
                submit(executor, queued, done, () -> job.run(doc));
            }
        }

        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

        failedByCollection.forEach((collectionName, failed) ->
                promote(mongoDatabase, collectionName, failed.get(), promoteNlpCollections));
        System.out.println("[nlp] Updated " + updated.get() + " fields, "
                + failedByCollection.values().stream().mapToInt(AtomicInteger::get).sum() + " failed");
    }

    private static TransliterationClient nlpClient() throws MalformedURLException {
        ObjectMapper mapper = new ObjectMapper()
                .registerModule(new JavaTimeModule())
                .registerModule(new SimpleModule(JAVA_TIME_MODULE_ID))
                .registerModule(new ProblemModule())
                .registerModule(new EntityApiModule());
        return TransliterationClient.of(mapper, URI.create(NLP_URL).toURL(), HttpClient.newHttpClient());
    }

    private static MongoCollection<Document> copyForNlp(MongoDatabase mongoDatabase, String collectionName,
                                                        EntityConfig config) {
        String nlpName = siblingName(collectionName, "_nlp");
        MongoCollection<Document> copy = mongoDatabase.getCollection(nlpName);
        copy.drop();
        mongoDatabase.getCollection(collectionName).aggregate(List.of(Aggregates.out(nlpName))).toCollection();
        System.out.println("[nlp] " + collectionName + " -> " + nlpName + ": copied " + copy.countDocuments()
                + " documents");
        for (Part part : PARTS) {
            if (config.requestFor(part) != null) {
                unsetEn(copy, part.entriesKey());
                unsetEn(copy, part.recordsKey());
            }
        }
        return copy;
    }

    private static void unsetEn(MongoCollection<Document> collection, String array) {
        var result = collection.updateMany(Filters.exists(array + ".0"), Updates.unset(array + ".$[].en"));
        System.out.println("[unset] " + collection.getNamespace().getCollectionName() + "." + array + ": matched="
                + result.getMatchedCount() + " modified=" + result.getModifiedCount());
    }

    private static void promote(MongoDatabase mongoDatabase, String collectionName, int failed, boolean promote) {
        String nlpName = siblingName(collectionName, "_nlp");
        MongoCollection<Document> live = mongoDatabase.getCollection(collectionName);
        MongoCollection<Document> copy = mongoDatabase.getCollection(nlpName);
        long liveCount = live.countDocuments();
        long nlpCount = copy.countDocuments();
        if (!promote) {
            System.out.println("[nlp] " + nlpName + " is left as it is: " + nlpCount + " documents, " + failed
                    + " failed requests");
        } else if (failed > 0 || liveCount != nlpCount) {
            System.out.println("[nlp] " + nlpName + " is NOT renamed to " + collectionName + ": " + failed
                    + " failed requests, " + nlpCount + " documents against " + liveCount);
        } else {
            copyIndexes(live, copy);
            copy.renameCollection(new MongoNamespace(mongoDatabase.getName(), collectionName),
                    new RenameCollectionOptions().dropTarget(true));
            System.out.println("[nlp] renamed " + nlpName + " to " + collectionName);
        }
    }

    private static void copyIndexes(MongoCollection<Document> from, MongoCollection<Document> to) {
        for (Document index : from.listIndexes()) {
            String indexName = index.getString("name");
            if ("_id_".equals(indexName)) {
                continue;
            }
            try {
                Number expireAfterSeconds = index.get("expireAfterSeconds", Number.class);
                to.createIndex(index.get("key", Document.class), new IndexOptions()
                        .name(indexName)
                        .unique(index.getBoolean("unique", false))
                        .sparse(index.getBoolean("sparse", false))
                        .partialFilterExpression(index.get("partialFilterExpression", Document.class))
                        .expireAfter(expireAfterSeconds == null ? null : expireAfterSeconds.longValue(),
                                TimeUnit.SECONDS));
            } catch (RuntimeException e) {
                System.out.println("[nlp] WARNING: index " + indexName + " of " + from.getNamespace().getCollectionName()
                        + " could not be copied, create it again by hand: " + e.getMessage());
            }
        }
    }

    private static void submit(ExecutorService executor, Semaphore queued, AtomicInteger done, Runnable work) {
        queued.acquireUninterruptibly();
        executor.execute(() -> {
            try {
                work.run();
            } finally {
                queued.release();
                int finished = done.incrementAndGet();
                if (finished % 500 == 0) {
                    System.out.println("[nlp] " + finished + " documents done");
                }
            }
        });
    }

    private static String onOff(boolean selected) {
        return selected ? "on" : "off";
    }

    private static String describe(EntityConfig config) {
        return "name=" + describe(config.name()) + ", address=" + describe(config.address());
    }

    private static String describe(RequestConfig request) {
        return request == null ? "off" : request.method() + " alternatives="
                + request.alternatives().stream().map(LanguageCode::get).toList() + " verify=" + request.verify();
    }
}
