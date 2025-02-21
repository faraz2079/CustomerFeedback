package de.fh_dortmund.inference.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import de.fh_dortmund.inference.domain.component.CustomMetricsBinder;
import de.fh_dortmund.inference.domain.request.InferenceRequest;
import de.fh_dortmund.inference.service.InferenceService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

import java.time.LocalDateTime;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@RestController
@RequestMapping("/api/v1")
public class InferenceServiceController {

	Logger logger = LogManager.getLogger(InferenceServiceController.class);

	@Autowired
	private InferenceService service;
	@Autowired
	private CustomMetricsBinder metrics;

	@PostMapping("/feedback")
	public ResponseEntity<String> analyseFeedback(@Valid @NotNull @RequestBody InferenceRequest request,
			HttpServletRequest httpReq) {
		try {
			request.setId((long) httpReq.getAttribute("requestId"));
			logger.info("Feedback received: " + " Request ID: " + request.getId() + " Review: " + request.getText());
			LocalDateTime startTime = LocalDateTime.now();
			metrics.incrementRequestCount();
			String sentiment = service.analyseFeedback(request);
			LocalDateTime endTime = LocalDateTime.now();
			logger.info("Request Processed: " + " Request ID: " + request.getId() + " Start Time: "
					+ startTime.toString() + " End Time: " + endTime.toString());
			logger.info("Overall Sentiment of the customer for the product: " + sentiment);
			return ResponseEntity.status(HttpStatus.OK).body(sentiment);
		} catch (Exception e) {
			logger.error("Exception occured while processing request: " + e.getMessage());
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
					.body("Exception occured while processing the request");
		}
	}
}
