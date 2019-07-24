#!/bin/bash

for i in {1..30}
do
	mv "$i.png" "$(($i-1)).png"
done
