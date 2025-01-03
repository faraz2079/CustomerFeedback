package de.fh_dortmund.inference.domain.component;

import org.springframework.stereotype.Component;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Metrics;
import io.micrometer.core.instrument.binder.MeterBinder;
import lombok.Getter;

@Component
@Getter
public class CustomMetricsBinder implements MeterBinder {

	private volatile float latency;
	private volatile float feedbackScore;
	private volatile float accuracy;
	private volatile float cpuUtilization;
	private volatile float powerConsumption;
	private final Counter requestCount;

	public CustomMetricsBinder(MeterRegistry meterRegistry) {
		meterRegistry.gauge("custom.latency", this, CustomMetricsBinder::getLatency);
		meterRegistry.gauge("custom.feedback_score", this, CustomMetricsBinder::getFeedbackScore);
		meterRegistry.gauge("custom.accuracy", this, CustomMetricsBinder::getAccuracy);
		meterRegistry.gauge("custom.cpu_utilization", this, CustomMetricsBinder::getCpuUtilization);
		meterRegistry.gauge("custom.power_consumption", this, CustomMetricsBinder::getPowerConsumption);
		this.requestCount = Counter.builder("request_count").description("Total count of requests")
				.register(Metrics.globalRegistry);
	}

	public void updateMetrics(float latency, float feedbackScore, float accuracy, float cpuUtilization,
			float powerConsumption) {
		this.latency = latency;
		this.feedbackScore = feedbackScore;
		this.accuracy = accuracy;
		this.cpuUtilization = cpuUtilization;
		this.powerConsumption = powerConsumption;
	}

	@Override
	public void bindTo(MeterRegistry registry) {
	}
	
	public void incrementRequestCount() {
        requestCount.increment();
    }
}
