library(caret)
library(ranger)

df_list <- lapply(list.files("data/LOSO/", full.names=T), read.csv, row.names=1)
an_df <- read.csv("data/storm_hours_outages.csv")
an_df$dateTime <- as.character(an_df$dateTime)

# Set up predictions data frame across all methods
pred_df <- data.frame(dateTime=an_df$dateTime,
                      countts=df_list[[1]]$countts[rownames(df_list[[1]]) %in% an_df$dateTime],
                      sid=(0:485)%/%18, hid=(0:485) %% 18)
pred_out_path <- "results/predictions.csv"

#################
## DEFINE LAGS ##
#################

lagDF <- function(d, lag) {
  if (lag == 0)
    return(d)
  
  d_lagged <- d[(lag+1):nrow(d),]
  for(j in 1:ncol(d)) {
    col <- d[,j]
    for(l in 1:lag) {
      col_lagged <- col[l:(l+nrow(d_lagged)-1)]
      d_lagged[,paste0(names(d)[j],"_l",l)] <- col_lagged
    }
  }
  
  return(d_lagged)
}

lagOutageDF <- function(d, lag) {
  countts <- d$countts[(lag+1):length(d$countts)]
  return(cbind(lagDF(d[,-ncol(d)], lag), countts))
}

lag <- 1

########################
## Poisson regression ##
########################

set.seed(42)
storm_pred <- c()
for(i in 0:26) {
  df <- lagOutageDF(df_list[[i+1]], lag)
  storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
  
  fit <- glm(countts ~., df[!(rownames(df) %in% storm_hours),], family="poisson")
  storm_pred <- c(storm_pred,
                  predict(fit, df[rownames(df) %in% storm_hours,], type="response"))
}
pred_df$glm <- storm_pred
write.csv(pred_df, pred_out_path, row.names=F) # Checkpoint

#########
## KNN ##
#########

# Search for k
set.seed(42)
hourly_cors <- c()
search_grid <- seq(2,300,2)
for(k in search_grid) {
  
  h_cor <- c()
  for(i in 0:26) {
    df <- lagOutageDF(df_list[[i+1]], lag)
    storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
    fit <- knnreg(countts ~., df[!(rownames(df) %in% storm_hours),], k=k)
    pred <- predict(fit, df[rownames(df) %in% storm_hours,])
    h_cor <- c(h_cor, cor(pred_df[pred_df$sid==i,]$countts, pred)^2)
  }
  hourly_cors <- c(hourly_cors, mean(h_cor))
  
}
# Save search
knn_search <- data.frame(k=search_grid, hourly_R2=hourly_cors)
write.csv(knn_search, "results/knn_search.csv", row.names=F)

# Predict using best k
storm_pred <- c()
for(i in 0:26) {
  df <- lagOutageDF(df_list[[i+1]], lag)
  storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
  fit <- knnreg(countts ~., df[!(rownames(df) %in% storm_hours),], k=74)
  storm_pred <- c(storm_pred,
                  predict(fit, df[rownames(df) %in% storm_hours,]))
}
pred_df$knn <- storm_pred
write.csv(pred_df, pred_out_path, row.names=F) # Checkpoint

########
## RF ##
########

# Select number of trees
set.seed(42)
hourly_cors <- c()
search_grid <- seq(50,1000,50)
for(ntree in search_grid) {
  
  h_cor <- c()
  for(i in 0:26) {
    df <- lagOutageDF(df_list[[i+1]], lag)
    storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
    fit <- ranger(countts ~., df[!(rownames(df) %in% storm_hours),],
                  num.trees=ntree)
    pred <- predict(fit, df[rownames(df) %in% storm_hours,])$predictions
    h_cor <- c(h_cor, cor(pred_df[pred_df$sid==i,]$countts, pred)^2)
  }
  hourly_cors <- c(hourly_cors, mean(h_cor))
  
}
# Save search
rf_search <- data.frame(ntree=search_grid, hourly_R2=hourly_cors)
write.csv(rf_search, "results/rf_search.csv", row.names=F)

# Predict using best number of trees
set.seed(42)
storm_pred <- c()
for(i in 0:26) {
  df <- lagOutageDF(df_list[[i+1]], lag)
  storm_hours <- an_df$dateTime[(i*18+1):((i+1)*18)]
  fit <- ranger(countts ~., df[!(rownames(df) %in% storm_hours),],
                num.trees=150)
  storm_pred <- c(storm_pred,
                  predict(fit, df[rownames(df) %in% storm_hours,])$predictions)
}
pred_df$rf <- storm_pred
write.csv(pred_df, pred_out_path, row.names=F) # Checkpoint

###################
## (import) LSTM ##
###################

pred_df$lstm <- read.csv("results/LSTM_pred.csv")$Predicted
write.csv(pred_df, pred_out_path, row.names=F) # Checkpoint

########################
## (import) Event-OPM ##
########################

tot_preds <- read.csv("results/EventOPM_pred.csv")$avgTS
pred_df$eopm <- rep(tot_preds/18, each=18)
write.csv(pred_df, pred_out_path, row.names=F) # Checkpoint
