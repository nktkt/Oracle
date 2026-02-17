package main

import (
	"flag"
	"fmt"
	"log"

	"oracle/internal/oracle"
)

func main() {
	var (
		dataPath string
		steps    int
		lag      int
		hidden   int
		epochs   int
		seed     int64
		lr       float64
	)

	flag.StringVar(&dataPath, "data", "data/sample.csv", "path to time-series data file")
	flag.IntVar(&steps, "steps", 5, "number of future points to predict")
	flag.IntVar(&lag, "lag", 6, "number of past points used for one prediction")
	flag.IntVar(&hidden, "hidden", 12, "hidden layer size")
	flag.IntVar(&epochs, "epochs", 1800, "training epochs")
	flag.Float64Var(&lr, "lr", 0.008, "learning rate")
	flag.Int64Var(&seed, "seed", 42, "random seed")
	flag.Parse()

	series, err := oracle.LoadSeriesFromFile(dataPath)
	if err != nil {
		log.Fatalf("failed to load data: %v", err)
	}

	result, err := oracle.Train(series, oracle.TrainConfig{
		Lag:          lag,
		Hidden:       hidden,
		Epochs:       epochs,
		LearningRate: lr,
		Seed:         seed,
	})
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}

	predictions, err := oracle.Forecast(result, series, steps)
	if err != nil {
		log.Fatalf("forecast failed: %v", err)
	}

	fmt.Println("Oracle - Future Forecast")
	fmt.Printf("Data points      : %d\n", len(series))
	fmt.Printf("Training MSE     : %.6f\n", result.MSE)
	fmt.Printf("Residual Std Dev : %.6f\n", result.ResidualStdDev)
	fmt.Printf("Last observed    : %.4f\n", series[len(series)-1])
	fmt.Println()

	for i, p := range predictions {
		low := p - 1.96*result.ResidualStdDev
		high := p + 1.96*result.ResidualStdDev
		fmt.Printf("t+%d -> %.4f  (95%% range: %.4f .. %.4f)\n", i+1, p, low, high)
	}
}
