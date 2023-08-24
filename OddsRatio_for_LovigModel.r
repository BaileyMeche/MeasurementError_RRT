#function run_deviance_withtrust: 
#  - runs model iterations, 
#  - runs a binary logistic regression between true responses and model responses 
#  - computes odds ratio on the BLR model 
run_deviance_withtrust <- function(p,q,c,A){
  set.seed(2018)

  collect <- c()
  for(iteration in c(1:m)){                 #run loop
    true_responses <- rbinom(n, 1, pi_x)    #generates true responses
    model_rec <- c()                        #lovig model with confusion no trust

    for(i in c(1:n)){
      random_device <- sample(3,1,c(q,p,1-p-q), replace=T)    # generates true values with prob pi_x or 1-pi_x
      true_resp <- true_responses[i]                          # iterates through each observation

      # if indirect question with probability q
      if(random_device ==1){
        if(true_resp == 1){ # sensitive = pi_x
          trust <- sample(c(0,1), 1, c(A, 1-A), replace=T)                      #trust or no trust, P(yes=1)=1-A
          if(trust ==1){model_rec[i] <- sample(c(0,1), 1, c(c, 1-c), replace=T)}#if resp =yes, record P(yes)=1-c
          if(trust ==0){model_rec[i] <- sample(c(0,1), 1, c(1-c,c), replace=T)}}         #yes trust: true response is yes, no=0 with prob 1-c
        if(true_resp == 0){   #sensitive = false = 1-pi_x
          model_rec[i] <- sample(c(0,1), 1, c(c, 1-c), replace=T)  }
      }
      # if direct question with probability p
      if(random_device ==2){
        if(true_resp == 1){  # sensitive = pi_x
          trust <- sample(c(1,0), 1, c(A, 1-A), replace=T)                      #trust or no trust, P(yes)=A
          if(trust ==1){model_rec[i] <- sample(c(0,1), 1, c(c, 1-c), replace=T)}#resp=yes: record P(yes)=1-c
          if(trust ==0){model_rec[i] <- sample(c(0,1), 1, c(1-c,c), replace=T)}}         #yes=1 with prob c
        if(true_resp == 0){
          model_rec[i] <-  sample(c(0,1), 1, c(1-c,c), replace=T)               #if true_resp = 1-pi_x
          }
      }
      if(random_device ==3){   # if unrelated question
         unrelated <- sample(c(0,1), 1, c(1-pi_y,pi_y), replace=T)              #adds in optionality to the unrelated branch
         if(unrelated==1) {model_rec[i] <-sample(c(0,1), 1, c(c, 1-c), replace=T)}
         if(unrelated==0) {model_rec[i] <-sample(c(1,0), 1, c(c, 1-c), replace=T)}
         #
      }
    }

    #now all responses through the model are recorded
    data <- data.frame(true_responses, model_rec)
    model <- glm(true_responses ~ model_rec , data=data, family="binomial")

    odds_ratio <- exp(as.numeric(model$coefficients[-1]))
    collect[iteration] <-  odds_ratio
  }

  return(collect)
}

#execute 
pi_x = 0.4
pi_y = 1/12
n=500
m=10000


p_q_cases = list(c(0.7,0),c(0.7,0.15),  c(0.7,0.3))
c_cases = c(0,0.05,0.1)
A_cases = c(1,0.95, 0.9)

rows_length = (length(p_q_cases)*length(c_cases)*length(A_cases))
results_dev <- data.frame(matrix(ncol = 5, nrow = rows_length))
collect <- c()

num<-1
for(p_q in p_q_cases){
  for(A in A_cases){
    for(c in c_cases){
      p_q_v <- unlist(p_q)
      dev <- run_deviance_withtrust(p_q_v[1], p_q_v[2],c,A)     #p,q,c,A

      #Check for outliers and pop off the values >1000
      if(identical(c(p_q_v[1], p_q_v[2], c), c(0.7, 0, 0)) ==TRUE) dev <- sort(dev, decreasing = T)[0:-5]

      collect <- append(collect,mean(dev))                            #collects all deviance results

      results_dev[num, ] <- c(p_q_v, p_q_v[1] +p_q_v[2],c,A)       #writes in a row with [(p,q),p+q,A,c] ; L=5   adds rows of p & q, total= 7*3*3 rows
      num<- num+1
    }
  }
}

# adds on a column with results
results_dev[,6] <- collect            

col_names <- c('p', 'q', 'p+q','c','A',  'OR')
colnames(results_dev) <- col_names
results_dev

plot(c(1:rows_length),unlist(results_dev[6]), type="o",col='blue', ylab='Predictability')  #, ylim=1, xlab="Column Names", ylab="Value") unlist(results_dev[3]),
title("OR")