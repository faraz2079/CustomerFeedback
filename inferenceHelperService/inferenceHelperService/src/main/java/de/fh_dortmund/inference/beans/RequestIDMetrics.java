package de.fh_dortmund.inference.beans;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;


@Data
@NoArgsConstructor
@AllArgsConstructor
public class RequestIDMetrics {
	
	private long requestID;
	private String podName;

}
