(ns pymc-intro
  (:require
   [scicloj.tempfiles.api :as tempfiles]
   [tablecloth.api :as tc]
   [fastmath.core :as math]
   [fastmath.random :as random]
   [tech.v3.datatype :as dtype]
   [tech.v3.datatype.functional :as fun]
   [aerial.hanami.common :as hc]
   [aerial.hanami.templates :as ht]
   [scicloj.kindly.v3.kind :as kind]
   [scicloj.kindly.v3.api :as kindly]
   [scicloj.clay.v2.api :as clay]
   [libpython-clj2.python :refer [py. py.. py.-] :as py]
   [scicloj.noj.v1.vis :as vis]
   [scicloj.noj.v1.vis.python :as vis.python]
   [libpython-clj2.require :refer [require-python]]))

;; https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html

(require-python '[builtins :as python]
                'operator
                '[arviz :as az]
                '[arviz.style :as az.style]
                '[pandas :as pd]
                '[matplotlib.pyplot :as plt]
                '[numpy :as np]
                '[numpy.random :as np.random]
                '[pymc :as pm])

(defn brackets [obj entry]
  (py. obj __getitem__ entry))

(def colon
  (python/slice nil nil))

(arviz.style/use "arviz-darkgrid")

(def random-seed 8927)
(def dataset-size 100)

(def true-parameter-values
  {:alpha 1
   :sigma 1
   :beta [1 2.5]})

(defn gen-dataset [{:keys [size random-seed
                           alpha sigma beta]}]
  (let [rng (random/rng :isaac random-seed)]
    (-> {:x1 (-> (dtype/make-reader :float32 size (rand))
                 dtype/clone)
         :x2 (-> (dtype/make-reader :float32 size (rand))
                 (fun/* 0.2)
                 dtype/clone)}
        tc/dataset
        (tc/add-column :y
                       #(-> (fun/+ alpha
                                   (fun/* (beta 0) (:x1 %))
                                   (fun/* (beta 1) (:x2 %))
                                   (fun/* sigma
                                          (dtype/make-reader
                                           :float32 size (rand))))
                            (dtype/clone))))))



(def dataset
  (gen-dataset (merge {:random-seed random-seed
                       :size dataset-size}
                      true-parameter-values)))

(->> [:x1 :x2]
     (mapv (fn [x]
             (-> dataset
                 (vis/hanami-plot ht/point-chart
                                  :X x)))))

pm/__version__


(def basic-model (pm/Model))

(py/with [_ basic-model]
         (let [{:keys [x1 x2 y]} (-> dataset
                                     (update-vals np/array))
               alpha (pm/Normal "alpha"
                                :mu 0
                                :sigma 10)
               beta (pm/Normal "bega"
                               :mu 0
                               :sigma 10
                               :shape 2)
               sigma (pm/HalfNormal "sigma"
                                    :sigma 1)
               mu (operator/add alpha
                                (operator/mul (brackets beta 0)
                                              x1)
                                (operator/mul (brackets beta 0)
                                              x2))
               y_obs (pm/Normal "y_obs"
                                :mu mu
                                :sigma sigma
                                :observed y)]))

(def idata
  (py/with [_ basic-model]
           (pm/sample)))


(-> idata
    (py.- posterior)
    (py.- alpha)
    (py. sel :draw (python/slice 0 4)))


(def slice-idata
  (py/with [_ basic-model]
           (let [step (pm/Slice)]
             (pm/sample 5000 :step step))))

(vis.python/pyplot
 #(az/plot_trace idata :combined true))


:bye
