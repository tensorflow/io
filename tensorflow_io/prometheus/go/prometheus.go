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
func Query(endpoint string, query string, sec int64, offset int64, key []int64, val []float64) int64 {
	client, err := api.NewClient(api.Config{
		Address: endpoint,
	})
	if err != nil {
		return -1
	}
	value, err := v1.NewAPI(client).Query(context.Background(), query, time.Unix(sec, 0))
	if err != nil {
		return -1
	}
	if m, ok := value.(model.Matrix); ok && m.Len() > 0 {
		index := int64(0)
		for index < int64(len(key)) && offset+index < int64(len(m[0].Values)) {
			v := m[0].Values[offset+index]
			key[index] = v.Timestamp.Unix()
			val[index] = float64(v.Value)
			index++
		}
		return index
	}
	return 0
}

func main() {
	key := make([]int64, 20, 20)
	val := make([]float64, 20, 20)
	sec := time.Now().Unix()
	fmt.Println(sec)
	returned := Query("http://localhost:9090", "coredns_dns_request_count_total[5m]", sec, 0, key, val)
	fmt.Println(returned)
	for i := range key {
		fmt.Printf("%d, %q, %v\n", i, model.TimeFromUnix(key[i]).Time(), val[i])
	}
}
