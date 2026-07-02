package com.bishwa.rag.config;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class RagConfig {
    private static final String OLLAMA_URL = "http://localhost:11434";

    @Bean
    public EmbeddingModel embeddingModel() {
        return new EmbeddingModel() {

            private static final int VECTOR_SIZE = 384;

            @Override
            public Response<Embedding> embed(String text) {
                return Response.from(Embedding.from(toVector(text)));
            }

            @Override
            public Response<List<Embedding>> embedAll(List<TextSegment> segments) {
                List<Embedding> embeddings = segments.stream()
                        .map(s -> Embedding.from(toVector(s.text())))
                        .toList();
                return Response.from(embeddings);
            }

            // Word-hashing: maps each word to a vector dimension via hash
            // Normalized so cosine similarity works correctly
            private float[] toVector(String text) {
                float[] vector = new float[VECTOR_SIZE];
                String[] words = text.toLowerCase()
                        .replaceAll("[^a-z0-9\\s]", "")
                        .split("\\s+");

                for (String word : words) {
                    if (word.isBlank()) continue;
                    int index = Math.abs(word.hashCode()) % VECTOR_SIZE;
                    vector[index] += 1.0f;

                    // also add bigram context with next char to reduce collisions
                    if (word.length() > 3) {
                        int index2 = Math.abs((word + word.charAt(0)).hashCode()) % VECTOR_SIZE;
                        vector[index2] += 0.5f;
                    }
                }

                // L2 normalize so cosine similarity = dot product
                float magnitude = 0;
                for (float v : vector) magnitude += v * v;
                magnitude = (float) Math.sqrt(magnitude);
                if (magnitude > 0) {
                    for (int i = 0; i < vector.length; i++) vector[i] /= magnitude;
                }
                return vector;
            }
        };
    }

    // ChatLanguageModel → auto-configured from langchain4j.anthropic.*
    // EmbeddingModel    → auto-configured from langchain4j.open-ai.embedding-model.*
    // You only need to define the store

    @Bean
    public EmbeddingStore<TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

//    @Bean
//    public ChatLanguageModel chatModel() {
//        return AnthropicChatModel.builder()
//                .apiKey(System.getenv("ANTHROPIC_API_KEY"))
//                .modelName("claude-sonnet-4-6")
//                .maxTokens(1024)
//                .build();
//    }

    @Bean
    public ChatLanguageModel chatModel() {
        return OllamaChatModel.builder()
                .baseUrl(OLLAMA_URL)
                .modelName("qwen2.5-coder:7b")  // ← just change this
                .temperature(0.3)
                .build();
    }
}