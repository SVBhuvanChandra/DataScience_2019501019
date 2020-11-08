transactions_data <- read.csv("transactions.csv")

names(transactions_data)

transactions_data$TID <- NULL

colnames(transactions_data) <- c("ItemList")

names(transactions_data)

write.csv(transactions_data,'ItemList.csv',quote = FALSE,row.names = TRUE)
install.packages("arules")
library(arules)
txn = read.transactions(file="ItemList.csv",rm.duplicates=TRUE,format="basket",sep=",",cols=1);

txn@itemInfo$labels <- gsub("\"","",txn@itemInfo$labels)
basket_rules <- apriori(txn,parameter = list(sup = 0.01, conf = 0.5, target = "rules"))

if (sessionInfo()['basepkgs']=="tm" | sessionInfo()['otherpkgs']=='tm'){
  detach(package:tm,unload=TRUE)
}
inspect(basket_rules)
