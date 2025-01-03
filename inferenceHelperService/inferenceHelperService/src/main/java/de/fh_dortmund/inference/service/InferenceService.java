package de.fh_dortmund.inference.service;

import de.fh_dortmund.inference.domain.request.InferenceRequest;

public interface InferenceService {

	public String analyseFeedback(InferenceRequest request) throws Exception;
}
