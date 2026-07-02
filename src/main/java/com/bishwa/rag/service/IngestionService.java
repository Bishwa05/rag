package com.bishwa.rag.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class IngestionService {

    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;

    @Value("${rag.chunk-size}")
    private int chunkSize;

    @Value("${rag.chunk-overlap}")
    private int chunkOverlap;

    public int ingest(MultipartFile file) throws IOException {
        String text = new String(file.getBytes(), StandardCharsets.UTF_8);
        String filename = file.getOriginalFilename();

        // 1. Wrap raw text in a Document
        Document document = Document.from(text);

        // 2. Split into overlapping chunks
        //    Overlap ensures context isn't lost at chunk boundaries
        DocumentSplitter splitter = DocumentSplitters.recursive(chunkSize, chunkOverlap);
        List<TextSegment> chunks = splitter.split(document);

        log.info("Ingesting '{}' → {} chunks", filename, chunks.size());

        // 3. Embed each chunk and store
        //    embedAll() runs in parallel internally
        List<Embedding> embeddings = embeddingModel.embedAll(chunks).content();
        embeddingStore.addAll(embeddings, chunks);

        log.info("Ingestion complete for '{}'", filename);
        return chunks.size();
    }
}