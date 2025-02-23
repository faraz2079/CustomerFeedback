package de.fh_dortmund.inference.domain.component;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import de.fh_dortmund.inference.beans.RequestIDMetrics;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Metrics;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.binder.MeterBinder;
import lombok.Getter;

@Component
@Getter
public class CustomMetricsBinder implements MeterBinder {

	private volatile long latency;
	private volatile float feedbackScore;
	private volatile float accuracy;
	private volatile float inferenceTime;
	private final Counter requestCount;
	private RequestIDMetrics metrics;

	public CustomMetricsBinder(MeterRegistry meterRegistry, @Value("${app.env:local}") String env) {
		this.metrics = new RequestIDMetrics(0, "Unknown");
		meterRegistry.gauge("latency", Tags.of("env", env), this, CustomMetricsBinder::getLatency);
		meterRegistry.gauge("feedback_score", Tags.of("env", env), this, CustomMetricsBinder::getFeedbackScore);
		meterRegistry.gauge("accuracy", Tags.of("env", env), this, CustomMetricsBinder::getAccuracy);
		meterRegistry.gauge("inference_time", Tags.of("env", env), this, CustomMetricsBinder::getInferenceTime);
		this.requestCount = Counter.builder("request_count")
				.tags("env", env, "requestID", String.valueOf(metrics.getRequestID()), "pod_name", metrics.getPodName())
				.description("Total count of requests").register(Metrics.globalRegistry);
	}

	public void updateMetrics(long latency, float feedbackScore, float accuracy, float inferenceTime,
			RequestIDMetrics metrics) {
		this.latency = latency;
		this.feedbackScore = feedbackScore;
		this.accuracy = accuracy;
		this.inferenceTime = inferenceTime;
		this.metrics.setRequestID(metrics.getRequestID());
		this.metrics.setPodName(metrics.getPodName());
	}

	@Override
	public void bindTo(MeterRegistry registry) {
	}

	public void incrementRequestCount() {
		requestCount.increment();
	}
}
