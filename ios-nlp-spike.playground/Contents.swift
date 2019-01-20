import Cocoa
import CreateML

//iOS Natural Language Processing in 6 lines of code
//https://towardsdatascience.com/ios-natural-language-processing-in-6-lines-of-code-36b08af7f440

//Change path to where the training data is
let data = try MLDataTable(contentsOf: URL(fileURLWithPath:"/Users/vu/w/learn/ios-nlp-spike/twitter-sanders-apple3.csv" ))

let (training_data,test_data) = data.randomSplit(by: 0.9)

//var str = "Hello, playground"

let classifier = try MLTextClassifier(trainingData: training_data, textColumn: "text", labelColumn: "class")


let metrics = classifier.evaluation(on: test_data)
metrics.classificationError

try classifier.prediction(from: "what is this government doing this for?")

//Change path to where you want to write the model data
//try classifier.write(toFile: "/Users/vu/w/learn/ios-nlp-spike/twitter-classifier")

