package oracle

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"
)

// LoadSeriesFromFile reads one time-series value per line (or CSV-like rows).
// For each row, the first parsable number is used.
func LoadSeriesFromFile(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	values := make([]float64, 0, 256)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		fields := splitFields(line)
		for _, field := range fields {
			v, parseErr := strconv.ParseFloat(field, 64)
			if parseErr == nil {
				values = append(values, v)
				break
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("no numeric values found in %s", path)
	}
	return values, nil
}

func splitFields(line string) []string {
	return strings.FieldsFunc(line, func(r rune) bool {
		return r == ',' || r == ';' || r == '\t' || unicode.IsSpace(r)
	})
}
