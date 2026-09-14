package org.localTranslation.tools;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * Reads Chinese words from {@link #CSV_PATH}, translates each to English by calling the real
 * ai-nlp-service (POST /api/ai-nlp/translations/batch) - the exact same TranslationEngine,
 * NameNormalizer and cleanTranslation logic used in production - and writes "original,translated"
 * back into the same file.
 *
 * <p>Requires ai-nlp-service running locally on {@link #AI_NLP_BASE_URL} (started separately -
 * this class never modifies ai-translation-service, only calls its already-running REST API).
 */
public class TranslationFromCSV {

    private static final Path CSV_PATH = Path.of("src/main/resources/chinese.csv");
    private static final String AI_NLP_BASE_URL = "http://localhost:8081";
    private static final String SOURCE_LANGUAGE = "zh";
    private static final String PROVIDER = "DEEPSEEK";

    static void main() throws Exception {
        List<String> lines = Files.readAllLines(CSV_PATH, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            System.out.println(CSV_PATH + " is empty - nothing to translate.");
            return;
        }

        List<String> words = new ArrayList<>();
        List<Integer> nonBlankIndices = new ArrayList<>();
        for (int i = 0; i < lines.size(); i++) {
            String word = firstColumn(lines.get(i));
            if (!word.isBlank()) {
                words.add(word);
                nonBlankIndices.add(i);
            }
        }

        List<String> translations = translateBatch(words);

        List<String> outputLines = new ArrayList<>(lines);
        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            String translated = translations.get(i);
            outputLines.set(nonBlankIndices.get(i), csvRow(word, translated));
            System.out.println(word + " -> " + translated);
        }

        Files.write(CSV_PATH, outputLines, StandardCharsets.UTF_8);
        System.out.println("Wrote " + outputLines.size() + " translations into " + CSV_PATH);
    }

    /** The Chinese word for a line - if the line already has "word,translation" from a prior run, keeps just the word. */
    private static String firstColumn(String line) {
        int comma = line.indexOf(',');
        return (comma < 0 ? line : line.substring(0, comma)).strip();
    }

    private static String csvRow(String original, String translated) {
        return csvField(original) + "," + csvField(translated);
    }

    private static String csvField(String value) {
        if (value == null) {
            return "";
        }
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    /** Calls ai-nlp-service's real batch endpoint - one HTTP call for every word, in request order. */
    private static List<String> translateBatch(List<String> words) throws IOException, InterruptedException {
        ObjectMapper mapper = new ObjectMapper();

        ArrayNode items = mapper.createArrayNode();
        for (String word : words) {
            ObjectNode item = mapper.createObjectNode();
            item.put("text", word);
            item.put("language", SOURCE_LANGUAGE);
            item.put("provider", PROVIDER);
            item.putNull("database");
            item.putNull("publisher");
            item.putNull("collection");
            items.add(item);
        }
        ObjectNode requestBody = mapper.createObjectNode();
        requestBody.set("items", items);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(AI_NLP_BASE_URL + "/api/ai-nlp/translations/batch"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody), StandardCharsets.UTF_8))
                .build();

        HttpResponse<String> response = HttpClient.newHttpClient()
                .send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() / 100 != 2) {
            throw new IOException("ai-nlp-service returned HTTP " + response.statusCode() + ": " + response.body());
        }

        JsonNode root = mapper.readTree(response.body());
        List<String> translations = new ArrayList<>();
        for (JsonNode item : root.path("items")) {
            JsonNode translation = item.path("translation");
            translations.add(translation.isNull() ? null : translation.asText());
        }
        return translations;
    }
}
