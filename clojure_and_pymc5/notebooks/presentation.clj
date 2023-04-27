^:kindly/hide-code?
(ns presentation
  (:require [proteins]))

^:kindly/hide-code?
(->> 4
     (iterate (fn [x]
                (-> x (* 1.25) int)))
     (take 15)
     (mapcat (fn [tune]
               [{:tune tune}
                (proteins/report {:tune tune})])))
