package de.fh_dortmund.inference.domain.response;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class InferenceResponse {

	private String sentiment;
	private float feedbackScore;
	private float accuracy;
	private float cpuUtilization;
	private float powerConsumption;

}
