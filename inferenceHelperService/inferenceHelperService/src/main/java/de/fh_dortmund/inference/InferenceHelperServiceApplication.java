package de.fh_dortmund.inference;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Scope;
import org.springframework.web.client.RestTemplate;

@SpringBootApplication
@ComponentScan
public class InferenceHelperServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(InferenceHelperServiceApplication.class, args);
	}

	@Bean(autowireCandidate = true)
	@Scope("singleton")
	RestTemplate createRestTemplate() {
		return new RestTemplate();
	}
}
