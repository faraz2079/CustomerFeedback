package de.fh_dortmund.inference.domain.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class InferenceResponse {

	@JsonProperty("sentiment")
	private String sentiment;
	@JsonProperty("feedback_score")
	private float feedbackScore;
	@JsonProperty("accuracy")
	private float accuracy;
	@JsonProperty("inference_time")
	private float inferenceTime;

}
