(ns util
  (:require [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [libpython-clj2.require :refer [require-python]]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]))

(require-python '[builtins :as python]
                '[numpy :as np])

(defn brackets [obj entry]
  (py. obj __getitem__ entry))

(def colon
  (python/slice nil nil))

(defn py-array->clj [py-array]
  (let [np-array (np/array py-array)]
    (-> np-array
        (py. flatten)
        dtype/->double-array
        (tensor/reshape (np/shape np-array)))))

(defn prep-dataset-for-cljs
  "Make sure a dataset can be serialized for cljs
  by converting dtyp-next buffers into vectors."
  [dataset]
  (-> dataset
      (update-vals vec)))

(defn tensor2d->np-matrix [tensor]
  (->> tensor
       (map np/array)
       python/list
       np/matrix))
