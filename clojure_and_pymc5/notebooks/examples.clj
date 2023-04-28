^:kindly/hide-code?
(ns example
  (:require [tablecloth.api :as tc]
            [fastmath.core :as math]
            [fastmath.random :as random]
            [tech.v3.datatype :as dtype]
            [tech.v3.dataset :as dataset]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as fun]
            [aerial.hanami.common :as hc]
            [aerial.hanami.templates :as ht]
            [scicloj.kindly.v3.kind :as kind]
            [scicloj.kindly.v3.api :as kindly]
            [scicloj.clay.v2.api :as clay]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [scicloj.noj.v1.vis :as vis]
            [scicloj.noj.v1.vis.python :as vis.python]
            [libpython-clj2.require :refer [require-python]]
            [util]
            [tablecloth.api :as tc])
  (:import java.lang.Math))

^{:kindly/hide-code? true
  :kind/hidden true}
(require-python '[builtins :as python]
                'operator
                '[arviz :as az]
                '[arviz.style :as az.style]
                '[numpy :as np]
                '[numpy.random :as np.random]
                '[pymc :as pm]
                '[Bio.PDB.PDBParser]
                '[Bio.PDB]
                '[Bio.PDB.Polypeptide]
                '[pytensor]
                '[pytensor.tensor :as pt]
                '[math])


;; # Bayesian Modeling and Computation in Python, Exercise 1M21

;; ## A probabilistic model

(def result
  (py/with [_ (pm/Model)]
           (let [;; I write about 6 functions every day.
                 functions (pm/Poisson "functions"
                                       :mu 6)
                 ;; The probability of a bug in a function is 1%.
                 p-bug 0.01
                 ;; One day, I had 3 bugs.
                 bugs (pm/Binomial "bugs"
                                   :n functions
                                   :p p-bug
                                   :observed 3)]
             ;; Draw some inference about the number
             ;; of functions I wrote that day.
             {:prior (pm/sample_prior_predictive)
              :inference-data (pm/sample :chains 1)})))

;; ## Assumed prior

(vis.python/pyplot
 #(-> result
      :prior
      (py.- prior)
      az/plot_density))

;; ## Inferred posterior

(vis.python/pyplot
 #(-> result
      :inference-data
      (py.- posterior)
      az/plot_density))


;; # Visualization using 3dmol

^:kindly/hide-code?
(defn shapes-view [style shapes]
  (kind/hiccup
   ['(fn [style shapes]
       [:div
        {:style (merge {:width "100%"
                        :height "500px"
                        :position "relative"}
                       style)
         :ref (fn [el]
                (let [config (clj->js
                              {:backgroundColor "0xffffff"})
                      viewer (.createViewer js/$3Dmol el #_config)]
                  (doseq [[shape-type shape-data] shapes]
                    (case shape-type
                      :sphere (.addSphere viewer (clj->js shape-data))
                      :cylinder (.addCylinder viewer (clj->js shape-data))))
                  (.zoomTo viewer)
                  (.render viewer)
                  (.zoom viewer 0.8 1000)))}
        ;; need to keep this symbol to let Clay infer the necessary dependency
        'three-d-mol])
    style
    (vec shapes)]))

;; ## Basic shapes

(shapes-view
 {:width 200 :height 200}
 [[:sphere {:center {:x 0
                     :y 0
                     :z 0}
            :radius 3 :color "green"}]
  [:cylinder {:start {:x 0 :y 10 :z 20}
              :end {:x 10 :y 0 :z 30}
              :radius 2.5 :color :teal :alpha 0.5}]])

;; ## An x-y-z dataset

(def points
  (-> (tensor/compute-tensor [3 100]
                             (fn [i j] (-> (rand)
                                           (* 10)
                                           (+ j)
                                           (* 3))))
      (tensor/slice 1)
      (->> (zipmap [:x :y :z]))
      tc/dataset))

points

;; ## Turning a dataset into cylinders

(defn dataset->cylinders [dataset options]
  (-> dataset
      (tc/rows :as-maps)
      (->> ((juxt identity rest))
           (apply mapv (fn [xyz0 xyz1]
                         [:cylinder (merge {:start xyz0
                                            :end xyz1}
                                           options)])))))

(-> points
    (dataset->cylinders {:color "purple"}))


;; ## Showing the cylinders
(-> points
    (dataset->cylinders {:color "purple"
                         :radius 2})
    (->> (shapes-view {:height 200
                       :width 200})))
