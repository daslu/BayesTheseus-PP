^:kindly/hide-code?
(ns presentation
  (:require [proteins]))

^:kindly/hide-code?
(->> 4
     (iterate (fn [x]
                (-> x (* 1.25) int)))
     (take 15)
     (mapcat (fn [tune]
               (println {:tune tune})
               (concat
                [{:tune tune}]
                (proteins/report {:tune tune
                                  :residues-limit 999})))))
