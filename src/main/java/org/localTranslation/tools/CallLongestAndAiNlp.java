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
        String database = "entity_tr_registry_2";

        AiProvider longestNameProvider = AiProvider.GEMINI;
        boolean longestOrganization = true;
        boolean longestPerson = false;

        AiProvider nlpProvider = AiProvider.GEMINI;

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
    private static final int DOCUMENTS_PER_REQUEST = 50;

    private static final Part NAME = new Part("names", "nameRecords", "original", "language", true);
    private static final Part ADDRESS = new Part("address", null, "address", "ln", false);
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

    private record Work(Document doc, String publisher, Map<String, List<Document>> arrays) {
    }

    private record Item(Document entry, String publisher) {
    }

    private record Mirror(Document entry, Document record) {
    }

    private record NlpJob(TransliterationClient client, AiProvider provider, String database, String collectionName,
                          EntityConfig config, MongoCollection<Document> target, AtomicInteger updated,
                          AtomicInteger repeated, AtomicInteger failed) {

        void run(List<Document> docs) {
            try {
                translate(docs);
            } catch (RuntimeException e) {
                if (docs.size() == 1) {
                    countFailure(e.toString());
                    return;
                }
                if (repeated.incrementAndGet() <= 5) {
                    System.out.println("[nlp] " + docs.size() + " documents failed together: " + e
                            + " - trying them one by one");
                }
                docs.forEach(doc -> run(List.of(doc)));
            }
        }

        private void translate(List<Document> docs) {
            List<Work> works = docs.stream()
                    .map(doc -> new Work(doc, resolvePublisher(doc), new LinkedHashMap<>()))
                    .toList();
            int sent = 0;
            for (Part part : PARTS) {
                RequestConfig request = config.requestFor(part);
                if (request != null) {
                    sent += translatePart(works, request, part);
                }
            }
            works.forEach(this::save);
            updated.addAndGet(sent);
        }

        private void save(Work work) {
            List<Bson> updates = work.arrays().entrySet().stream()
                    .filter(array -> !array.getValue().isEmpty())
                    .map(array -> Updates.set(array.getKey(), array.getValue()))
                    .toList();
            if (!updates.isEmpty()) {
                target.updateOne(Filters.eq("_id", work.doc().get("_id")), Updates.combine(updates));
            }
        }

        private int translatePart(List<Work> works, RequestConfig request, Part part) {
            List<Item> toSend = new ArrayList<>();
            List<Mirror> mirrors = new ArrayList<>();
            for (Work work : works) {
                List<Document> entries = arrayOf(work.doc(), part.entriesKey());
                List<Document> records = arrayOf(work.doc(), part.recordsKey());
                Map<String, Document> entryByText = entries.stream()
                        .filter(entry -> entry.getString(part.textKey()) != null)
                        .collect(Collectors.toMap(entry -> entry.getString(part.textKey()), Function.identity(),
                                (first, second) -> first));
                for (Document entry : entries) {
                    if (isSendable(entry, part)) {
                        toSend.add(new Item(entry, work.publisher()));
                    }
                }
                for (Document record : records) {
                    String text = record.getString(part.textKey());
                    if (text == null || sourceLanguage(record, part) == null) {
                        continue;
                    }
                    Document sameEntry = entryByText.get(text);
                    if (sameEntry != null) {
                        mirrors.add(new Mirror(sameEntry, record));
                    } else if (!text.isBlank()) {
                        toSend.add(new Item(record, work.publisher()));
                    }
                }
                work.arrays().put(part.entriesKey(), entries);
                if (part.recordsKey() != null) {
                    work.arrays().put(part.recordsKey(), records);
                }
            }
            int sent = send(toSend, request, part);
            mirrors.forEach(mirror -> copyAnswer(mirror.entry(), mirror.record()));
            return sent;
        }

        private int send(List<Item> items, RequestConfig request, Part part) {
            if (items.isEmpty()) {
                return 0;
            }
            List<TransliterationRequest> requests = items.stream()
                    .map(item -> new TransliterationRequest(item.entry().getString(part.textKey()),
                            sourceLanguage(item.entry(), part), LanguageCode.ENGLISH, request.alternatives(),
                            request.verify(), request.method(), provider, database, item.publisher(),
                            collectionName))
                    .toList();
            Either<?, List<TransliterationResponse>> result = client.transliterate(requests);
            if (result.isLeft()) {
                throw new IllegalStateException(collectionName + ": " + result.getLeft());
            }
            List<TransliterationResponse> responses = result.get();
            if (responses.size() != items.size()) {
                throw new IllegalStateException(collectionName + ": " + items.size() + " requests but "
                        + responses.size() + " answers");
            }
            for (int i = 0; i < items.size(); i++) {
                Document entry = items.get(i).entry();
                TransliterationResponse response = responses.get(i);
                entry.put("en", response.getText());
                response.getCodeOpt().ifPresent(code -> entry.put(part.languageKey(), code.get()));
            }
            return items.size();
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
            System.out.println("[publisher] collection has multiple publishers " + publishers
                    + "; the service translates every document under its own publisher, the first one ("
                    + publishers.getFirst() + ") is only the fallback for documents without one");
        }
        return publishers.isEmpty() ? null : publishers.getFirst();
    }

    private static String resolvePublisher(Document doc) {
        Document metadata = doc.get("metadata", Document.class);
        List<String> publishers = metadata == null ? List.of() : metadata.getList("publisher", String.class, List.of());
        return publishers.isEmpty() ? null : publishers.getFirst();
    }

    private static List<Document> arrayOf(Document doc, String array) {
        return array == null ? List.of() : doc.getList(array, Document.class, List.of());
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
        AtomicInteger repeated = new AtomicInteger();
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
            NlpJob job = new NlpJob(client, provider, database, collectionName, config, copy, updated, repeated,
                    new AtomicInteger());
            failedByCollection.put(collectionName, job.failed());
            List<Document> group = new ArrayList<>();
            for (Document doc : copy.find().noCursorTimeout(true).batchSize(500)) {
                group.add(doc);
                if (group.size() == DOCUMENTS_PER_REQUEST) {
                    submit(executor, queued, done, job, group);
                    group = new ArrayList<>();
                }
            }
            if (!group.isEmpty()) {
                submit(executor, queued, done, job, group);
            }
        }

        executor.shutdown();
        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);

        failedByCollection.forEach((collectionName, failed) ->
                promote(mongoDatabase, collectionName, failed.get(), promoteNlpCollections));
        System.out.println("[nlp] Updated " + updated.get() + " fields, "
                + failedByCollection.values().stream().mapToInt(AtomicInteger::get).sum() + " failed, "
                + repeated.get() + " groups of documents were repeated one by one");
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
                if (part.recordsKey() != null) {
                    unsetEn(copy, part.recordsKey());
                }
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

    private static void submit(ExecutorService executor, Semaphore queued, AtomicInteger done, NlpJob job,
                               List<Document> docs) {
        queued.acquireUninterruptibly();
        executor.execute(() -> {
            try {
                job.run(docs);
            } finally {
                queued.release();
                int finished = done.addAndGet(docs.size());
                if (finished / 500 > (finished - docs.size()) / 500) {
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
