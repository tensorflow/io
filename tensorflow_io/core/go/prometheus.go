package main

import "C"

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/api"
	"github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
)

//export Query
func Query(endpoint string, query string, ts int64, timestamp []int64, value []float64) int {
	client, err := api.NewClient(api.Config{
		Address: endpoint,
	})
	if err != nil {
		return -1
	}
	v, err := v1.NewAPI(client).Query(context.Background(), query, time.Unix(ts, 0))
	if err != nil {
		return -1
	}
	if m, ok := v.(model.Matrix); ok && m.Len() > 0 {
		if len(timestamp) >= len(m[0].Values) && len(value) == len(m[0].Values) {

			for i := 0; i < len(m[0].Values); i++ {
				v := m[0].Values[i]
				timestamp[i] = int64(v.Timestamp)
				value[i] = float64(v.Value)
			}
		}

		return len(m[0].Values)
	}
	return 0
}

func main() {
	key := make([]int64, 20, 20)
	val := make([]float64, 20, 20)
	sec := time.Now().Unix()
	fmt.Println(sec)
	returned := Query("http://localhost:9090", "coredns_dns_request_count_total[5m]", sec, key, val)
	fmt.Println(returned)
	for i := range key {
		fmt.Printf("%d, %q, %v\n", i, model.TimeFromUnix(key[i]).Time(), val[i])
	}
}
