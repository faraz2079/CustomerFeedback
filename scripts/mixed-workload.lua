wrk.method = "POST"
wrk.path = "/feedback/analyse"
wrk.headers["Content-Type"] = "application/json"

local feedbacks = {
  { text = "This product is amazing and exceeded all my expectations.", stars = 5 },
  { text = "Very disappointed. Totally not worth the money.", stars = 1 },
  { text = "It works okay, nothing special.", stars = 3 },
  { text = "Good value for the price. Would recommend.", stars = 4 },
  { text = "Terrible experience. It broke on the first day.", stars = 1 },
  { text = "Excellent service and great quality!", stars = 5 }
}

math.randomseed(os.time())

request = function()
  local f = feedbacks[math.random(1, #feedbacks)]
  local body = string.format('{"text":"%s", "stars": %d}', f.text, f.stars)
  wrk.body = body
  return wrk.format(nil, wrk.path, wrk.headers, wrk.body)
end
