package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"oracle/internal/oracle"
)

type ForecastPoint struct {
	Step       int     `json:"step"`
	Prediction float64 `json:"prediction"`
	Low95      float64 `json:"low_95"`
	High95     float64 `json:"high_95"`
}

type ValidationPayload struct {
	Count int     `json:"count"`
	MAE   float64 `json:"mae"`
	RMSE  float64 `json:"rmse"`
	MAPE  float64 `json:"mape"`
}

type OutputPayload struct {
	DataPoints      int                `json:"data_points"`
	Lag             int                `json:"lag"`
	TrainingMSE     float64            `json:"training_mse"`
	ResidualStdDev  float64            `json:"residual_std_dev"`
	LastObserved    float64            `json:"last_observed"`
	ModelLoadedFrom string             `json:"model_loaded_from,omitempty"`
	ModelSavedTo    string             `json:"model_saved_to,omitempty"`
	Validation      *ValidationPayload `json:"validation,omitempty"`
	Forecast        []ForecastPoint    `json:"forecast"`
	ForecastCSVPath string             `json:"forecast_csv_path,omitempty"`
}

func main() {
	var (
		dataPath      string
		outPath       string
		outputFormat  string
		saveModelPath string
		loadModelPath string
		steps         int
		lag           int
		hidden        int
		epochs        int
		holdout       int
		seed          int64
		lr            float64
	)

	flag.StringVar(&dataPath, "data", "data/sample.csv", "path to time-series data file")
	flag.StringVar(&outPath, "out", "", "optional CSV output path for forecasts")
	flag.StringVar(&outputFormat, "format", "text", "output format: text or json")
	flag.StringVar(&saveModelPath, "save-model", "", "optional path to save trained model JSON")
	flag.StringVar(&loadModelPath, "load-model", "", "optional path to load model JSON and skip training")
	flag.IntVar(&steps, "steps", 5, "number of future points to predict")
	flag.IntVar(&lag, "lag", 6, "number of past points used for one prediction")
	flag.IntVar(&hidden, "hidden", 12, "hidden layer size")
	flag.IntVar(&epochs, "epochs", 1800, "training epochs")
	flag.IntVar(&holdout, "holdout", 0, "number of tail points for one-step holdout validation (0 disables)")
	flag.Float64Var(&lr, "lr", 0.008, "learning rate")
	flag.Int64Var(&seed, "seed", 42, "random seed")
	flag.Parse()

	outputFormat = strings.ToLower(strings.TrimSpace(outputFormat))
	if outputFormat != "text" && outputFormat != "json" {
		log.Fatalf("invalid -format: %q (use text or json)", outputFormat)
	}

	series, err := oracle.LoadSeriesFromFile(dataPath)
	if err != nil {
		log.Fatalf("failed to load data: %v", err)
	}

	cfg := oracle.TrainConfig{
		Lag:          lag,
		Hidden:       hidden,
		Epochs:       epochs,
		LearningRate: lr,
		Seed:         seed,
	}

	var (
		result      *oracle.TrainResult
		validation  *oracle.ValidationMetrics
		modelLoaded string
		modelSaved  string
	)

	if loadModelPath != "" {
		result, err = oracle.LoadModel(loadModelPath)
		if err != nil {
			log.Fatalf("loading model failed: %v", err)
		}
		modelLoaded = loadModelPath

		if holdout > 0 {
			metrics, validateErr := oracle.Validate(result, series, holdout)
			if validateErr != nil {
				log.Fatalf("validation failed: %v", validateErr)
			}
			validation = &metrics
		}
	} else {
		trainSeries := series
		if holdout > 0 {
			if holdout >= len(series) {
				log.Fatalf("invalid -holdout: %d (must be smaller than data length %d)", holdout, len(series))
			}
			trainSeries = series[:len(series)-holdout]
		}

		result, err = oracle.Train(trainSeries, cfg)
		if err != nil {
			log.Fatalf("training failed: %v", err)
		}

		if holdout > 0 {
			metrics, validateErr := oracle.Validate(result, series, holdout)
			if validateErr != nil {
				log.Fatalf("validation failed: %v", validateErr)
			}
			validation = &metrics

			// Retrain on full data so future forecasts use all observed points.
			result, err = oracle.Train(series, cfg)
			if err != nil {
				log.Fatalf("full-data retraining failed: %v", err)
			}
		}
	}

	if saveModelPath != "" {
		if err := oracle.SaveModel(saveModelPath, result); err != nil {
			log.Fatalf("saving model failed: %v", err)
		}
		modelSaved = saveModelPath
	}

	predictions, err := oracle.Forecast(result, series, steps)
	if err != nil {
		log.Fatalf("forecast failed: %v", err)
	}

	points := buildForecastPoints(predictions, result.ResidualStdDev)

	if outPath != "" {
		if err := writeForecastCSV(outPath, points); err != nil {
			log.Fatalf("failed writing forecast CSV: %v", err)
		}
	}

	if outputFormat == "json" {
		payload := OutputPayload{
			DataPoints:      len(series),
			Lag:             result.Lag,
			TrainingMSE:     result.MSE,
			ResidualStdDev:  result.ResidualStdDev,
			LastObserved:    series[len(series)-1],
			ModelLoadedFrom: modelLoaded,
			ModelSavedTo:    modelSaved,
			Forecast:        points,
			ForecastCSVPath: outPath,
		}
		if validation != nil {
			payload.Validation = &ValidationPayload{
				Count: validation.Count,
				MAE:   validation.MAE,
				RMSE:  validation.RMSE,
				MAPE:  validation.MAPE,
			}
		}
		body, marshalErr := json.MarshalIndent(payload, "", "  ")
		if marshalErr != nil {
			log.Fatalf("failed to encode json output: %v", marshalErr)
		}
		fmt.Println(string(body))
		return
	}

	fmt.Println("Oracle - Future Forecast")
	fmt.Printf("Data points      : %d\n", len(series))
	fmt.Printf("Lag              : %d\n", result.Lag)
	fmt.Printf("Training MSE     : %.6f\n", result.MSE)
	fmt.Printf("Residual Std Dev : %.6f\n", result.ResidualStdDev)
	fmt.Printf("Last observed    : %.4f\n", series[len(series)-1])
	if modelLoaded != "" {
		fmt.Printf("Model loaded     : %s\n", modelLoaded)
	}
	if modelSaved != "" {
		fmt.Printf("Model saved      : %s\n", modelSaved)
	}
	if validation != nil {
		fmt.Println()
		fmt.Printf("Holdout points   : %d\n", validation.Count)
		fmt.Printf("Validation MAE   : %.6f\n", validation.MAE)
		fmt.Printf("Validation RMSE  : %.6f\n", validation.RMSE)
		fmt.Printf("Validation MAPE  : %.4f%%\n", validation.MAPE)
	}
	fmt.Println()

	for _, p := range points {
		fmt.Printf("t+%d -> %.4f  (95%% range: %.4f .. %.4f)\n", p.Step, p.Prediction, p.Low95, p.High95)
	}
	if outPath != "" {
		fmt.Printf("\nSaved forecast CSV: %s\n", outPath)
	}
}

func buildForecastPoints(predictions []float64, residualStdDev float64) []ForecastPoint {
	points := make([]ForecastPoint, 0, len(predictions))
	delta := 1.96 * residualStdDev
	for i, p := range predictions {
		points = append(points, ForecastPoint{
			Step:       i + 1,
			Prediction: p,
			Low95:      p - delta,
			High95:     p + delta,
		})
	}
	return points
}

func writeForecastCSV(path string, points []ForecastPoint) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	w := csv.NewWriter(file)
	if err := w.Write([]string{"step", "prediction", "low_95", "high_95"}); err != nil {
		return err
	}

	for _, p := range points {
		row := []string{
			strconv.Itoa(p.Step),
			fmt.Sprintf("%.6f", p.Prediction),
			fmt.Sprintf("%.6f", p.Low95),
			fmt.Sprintf("%.6f", p.High95),
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	w.Flush()
	return w.Error()
}
