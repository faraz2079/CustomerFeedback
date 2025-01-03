package de.fh_dortmund.inference.domain.component;

import org.springframework.stereotype.Component;

import io.micrometer.core.instrument.MeterRegistry;
import lombok.Getter;

@Component
@Getter
public class MetricsBinder {

	private volatile float latency;
	private volatile float feedbackScore;
	private volatile float accuracy;
	private volatile float cpuUtilization;
	private volatile float powerConsumption;

	public MetricsBinder(MeterRegistry meterRegistry) {
		meterRegistry.gauge("custom.latency", this, MetricsBinder::getLatency);
		meterRegistry.gauge("custom.feedback_score", this, MetricsBinder::getFeedbackScore);
		meterRegistry.gauge("custom.accuracy", this, MetricsBinder::getAccuracy);
		meterRegistry.gauge("custom.cpu_utilization", this, MetricsBinder::getCpuUtilization);
		meterRegistry.gauge("custom.power_consumption", this, MetricsBinder::getPowerConsumption);
	}

	public void updateMetrics(float latency, float feedbackScore, float accuracy, float cpuUtilization,
			float powerConsumption) {
		this.latency = latency;
		this.feedbackScore = feedbackScore;
		this.accuracy = accuracy;
		this.cpuUtilization = cpuUtilization;
		this.powerConsumption = powerConsumption;
	}
}
