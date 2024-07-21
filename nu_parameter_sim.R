values <- seq(from=430, to=435, by=1)
avg_flips <- c()

for(i in 1:length(values)){
  x <- values[i]
  flips <- c()
  for(j in 1:1000){
    v <- 2 / choose(x,2)
    flips[j] <- sum(sample(c(0,1), size=46656, replace=T, prob=c(1-v, v)))
  }
  avg_flips[i] <- mean(flips)
}

print(avg_flips)
