package de.fh_dortmund.inference.domain.component;

import java.util.concurrent.atomic.AtomicLong;

import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class RequestIdInterceptor implements HandlerInterceptor {

	private static AtomicLong reqId = new AtomicLong(0);

	@Override
	public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
		long id = reqId.incrementAndGet();
		request.setAttribute("requestId", id);
		return true;
	}

}
