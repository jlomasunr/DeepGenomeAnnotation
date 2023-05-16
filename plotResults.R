library(ggplot2)
library(data.table)

setwd("/Users/jslomas/Box/STAT_760/DeepGenomeAnnotation")
result <- as.data.table(read.csv("results.csv"))
result <-  melt(result,id.vars = "Seq",variable.name="Class", value.name = "Percent")
result$Seq <- factor(result$Seq, levels=c("Train",
                                          "Test 92",
                                          "Test 115",
                                          "Test 143",
                                          "Test 155",
                                          "Test 170"))
ggplot(result, aes(x = Seq, y=Percent, fill=Class)) +
  geom_bar(position="dodge", stat="identity") +
  theme(axis.text.x = element_text(angle = -45, vjust = 0.3, hjust=0.2)) +
  xlab("Sequence") +
  ylab("Correctly Classified Bases (%)")
