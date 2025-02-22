package de.fh_dortmund.inference;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Scope;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import de.fh_dortmund.inference.domain.component.RequestIdInterceptor;

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

	@Bean
	WebMvcConfigurer webMvcConfigurer(RequestIdInterceptor reqInterceptor) {
		return new WebMvcConfigurer() {
			@Override
			public void addInterceptors(InterceptorRegistry registry) {
				registry.addInterceptor(reqInterceptor).addPathPatterns("/api/**");
			}

		};
	}
}
