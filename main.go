package main

import (
	"fmt"
	"lberg/tinyGoGrad/grad"
)

func main() {
	x1 := grad.NewInputParameter(4.0)
	x2 := grad.NewInputParameter(3.0)

	y1 := x1.Mul(2)
	y2 := x2.Mul(2)

	w1 := y1.Plus(y2)
	w2 := w1.Rec()
	z := w2.Mul(x1)

	z.Backward()
	fmt.Printf("grads: x1: %f, x2: %f", x1.Grad, x2.Grad)
}
