package main

import "C"

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/api"
	"github.com/prometheus/client_golang/api/prometheus/v1"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/common/model"
	"github.com/prometheus/prom2json"
)

//export QuerySpecs
func QuerySpecs(endpoint string, query string, timestamp int64, jobs [][]byte, instances [][]byte, names [][]byte) int {
	client, err := api.NewClient(api.Config{
		Address: endpoint,
	})
	if err != nil {
		return -1
	}
	r := v1.Range{
		Start: time.Unix(timestamp/1000, (timestamp%1000)*1000000),
		End:   time.Unix(timestamp/1000, (timestamp%1000)*1000000),
		Step:  time.Second,
	}
	v, err := v1.NewAPI(client).QueryRange(context.Background(), query, r)
	if err != nil {
		return -1
	}
	if m, ok := v.(model.Matrix); ok && m.Len() > 0 {
		for index := 0; index < len(m) && index < len(jobs) && index < len(instances) && index < len(names); index++ {
			job := string(m[index].Metric["job"])
			if len(job) >= cap(jobs[index]) {
				job = job[:cap(jobs[index])]
			}
			copy(jobs[index], []byte(job))
			instance := string(m[index].Metric["instance"])
			if len(instance) >= cap(instances[index]) {
				instance = instance[:cap(instances[index])]
			}
			copy(instances[index], []byte(instance))
			name := string(m[index].Metric["__name__"])
			if len(name) >= cap(names[index]) {
				name = name[:cap(names[index])]
			}
			copy(names[index], []byte(name))
		}
		return len(m)
	}
	return 0

}

//export QueryRange
func QueryRange(endpoint string, query string, start int64, end int64, job string, instance string, name string, timestamp []int64, value []float64) int {
	client, err := api.NewClient(api.Config{
		Address: endpoint,
	})
	if err != nil {
		return -1
	}
	r := v1.Range{
		Start: time.Unix(start/1000, (start%1000)*1000000),
		End:   time.Unix(end/1000, (end%1000)*1000000),
		Step:  time.Second,
	}
	v, err := v1.NewAPI(client).QueryRange(context.Background(), query, r)
	if err != nil {
		return -1
	}
	if m, ok := v.(model.Matrix); ok && m.Len() > 0 {
		for index := 0; index < len(m); index++ {
			if m[index].Metric["job"] == model.LabelValue(job) && m[index].Metric["instance"] == model.LabelValue(instance) && m[index].Metric["__name__"] == model.LabelValue(name) {
				for i := 0; i < len(m[index].Values) && i < len(timestamp) && i < len(value); i++ {
					v := m[index].Values[i]
					timestamp[i] = int64(v.Timestamp)
					value[i] = float64(v.Value)
				}

				return len(m[index].Values)
			}
		}
	}
	return 0
}

//export Scrape
func Scrape(endpoint string, metric string, value []float64) int {
	skipServerCertCheck := true
	transport := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: skipServerCertCheck},
	}

	mfChan := make(chan *dto.MetricFamily, 1024)

	err := prom2json.FetchMetricFamilies(endpoint, mfChan, transport)
	if err != nil {
		return -1
	}
	for mf := range mfChan {
		for _, m := range mf.Metric {
			if metric == mf.GetName() {
				if len(value) > 0 {
					value[0] = getValue(m)
					return 0
				}
			}
		}
	}
	return -1

}
func getValue(m *dto.Metric) float64 {
	if m.Gauge != nil {
		return m.GetGauge().GetValue()
	}
	if m.Counter != nil {
		return m.GetCounter().GetValue()
	}
	if m.Untyped != nil {
		return m.GetUntyped().GetValue()
	}
	return 0.
}

func main() {
	key := make([]int64, 20, 20)
	val := make([]float64, 20, 20)
	end := time.Now().Unix() * 1000
	start := end - 5*1000
	fmt.Println(start, end)
	returned := QueryRange("http://localhost:9090", "coredns_dns_request_count_total", start, end, "", "", "", key, val)
	fmt.Println(returned)
	for i := range key {
		fmt.Printf("%d, %q, %v\n", i, model.TimeFromUnix(key[i]).Time(), val[i])
	}
}
