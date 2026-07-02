package com.bishwa.rag.controller;

import com.bishwa.rag.model.IngestResponse;
import com.bishwa.rag.model.QueryRequest;
import com.bishwa.rag.service.IngestionService;
import com.bishwa.rag.service.QueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class DocumentController {

    private final IngestionService ingestionService;
    private final QueryService queryService;

    // Step 1: Upload a text/PDF file
    @PostMapping("/ingest")
    public ResponseEntity<IngestResponse> ingest(@RequestParam("file") MultipartFile file)
            throws IOException {
        int chunks = ingestionService.ingest(file);
        return ResponseEntity.ok(new IngestResponse(file.getOriginalFilename(), chunks));
    }

    // Step 2: Ask a question about the ingested document
    @PostMapping("/query")
    public ResponseEntity<String> query(@RequestBody QueryRequest request) {
        String answer = queryService.query(request.question());
        return ResponseEntity.ok(answer);
    }
}
