package main

import (
	"fmt"
	"lberg/tinyGoGrad/nn"
	"lberg/tinyGoGrad/optim"
)

func main() {
	sgd := optim.NewSGD(0.1)

	w := nn.NewParameter(1.0)
	sgd.AddParameter(w)
	b := nn.NewParameter(0.0)
	sgd.AddParameter(b)

	inputVal := 1.0
	targetVal := 2.0

	for range 100 {
		sgd.ZeroGrad()
		y := nn.NewParameter(inputVal).Mul(w).Plus(b)
		loss := nn.Pow(y.Minus(targetVal), 2)
		loss.Backward()
		sgd.Step()
	}

	yFin := nn.NewParameter(inputVal).Mul(w).Plus(b)

	fmt.Printf("val: w: %f, b: %f, pred: %f", w.Val, b.Val, yFin.Val)
}
