package de.fh_dortmund.inference.domain.request;

import java.io.Serializable;

import com.fasterxml.jackson.annotation.JsonIgnore;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class InferenceRequest implements Serializable {

	private static final long serialVersionUID = -2512078060170749966L;
	@JsonIgnore(value = true)
	private long id;
	@NotNull(message = "No text present in text field")
	@NotBlank(message = "Text field cannot be empty")
	private String text;
	@Min(value = 1, message = "Minimum stars should be 1")
	@Max(value = 5, message = "Maximum stars should be 5")
	private int stars;

}
