package de.fh_dortmund.inference;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
/*import org.apache.hc.core5.http.ConnectionReuseStrategy;
import org.apache.hc.core5.http.HttpRequest;
import org.apache.hc.core5.http.HttpResponse;
import org.apache.hc.core5.http.protocol.HttpContext;*/
import org.apache.hc.core5.util.TimeValue;

import de.fh_dortmund.inference.domain.component.RequestIdInterceptor;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;

@SpringBootApplication
@ComponentScan
public class InferenceHelperServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(InferenceHelperServiceApplication.class, args);
	}

	/*
	 * @Bean(autowireCandidate = true) RestTemplate createRestTemplate() {
	 * HttpComponentsClientHttpRequestFactory factory = new
	 * HttpComponentsClientHttpRequestFactory(
	 * HttpClients.custom().setConnectionReuseStrategy(new ConnectionReuseStrategy()
	 * {
	 * 
	 * @Override public boolean keepAlive(HttpRequest request, HttpResponse
	 * response, HttpContext context) { return false; } }).build()); return new
	 * RestTemplate(factory); }
	 */

	@Bean
	CloseableHttpClient pooledHttpClient() {
		PoolingHttpClientConnectionManager connectionManager = new PoolingHttpClientConnectionManager();
		connectionManager.setMaxTotal(30);
		return HttpClients.custom().setConnectionManager(connectionManager).evictExpiredConnections()
				.evictIdleConnections(TimeValue.ofSeconds(3)).build();
	}

	@Bean
	RestTemplate createRestTemplate(CloseableHttpClient pooledHttpClient, RestTemplateBuilder restTemplateBuilder) {
		return restTemplateBuilder.requestFactory(() -> new HttpComponentsClientHttpRequestFactory(pooledHttpClient))
				.build();
	}

	@Bean
	WebMvcConfigurer webMvcConfigurer(RequestIdInterceptor reqInterceptor) {
		return new WebMvcConfigurer() {
			@Override
			public void addInterceptors(InterceptorRegistry registry) {
				registry.addInterceptor(reqInterceptor).addPathPatterns("/api/**");
			}

		};
	}
	
	@Bean
	OtlpHttpSpanExporter otlpHttpSpanExporter(@Value("${tracing.url}") String url) {
		return OtlpHttpSpanExporter.builder().setEndpoint(url).build();
	}
}
