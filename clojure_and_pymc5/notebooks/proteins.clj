(ns proteins
  (:require
   [scicloj.tempfiles.api :as tempfiles]
   [tablecloth.api :as tc]
   [fastmath.core :as math]
   [fastmath.random :as random]
   [tech.v3.datatype :as dtype]
   [tech.v3.dataset :as dataset]
   [tech.v3.dataset.tensor :as dataset.tensor]
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
   [libpython-clj2.require :refer [require-python]])
  (:import java.lang.Math))

;; https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html

(require-python '[builtins :as python]
                'operator
                '[arviz :as az]
                '[arviz.style :as az.style]
                '[pandas :as pd]
                '[matplotlib.pyplot :as plt]
                '[numpy :as np]
                '[numpy.random :as np.random]
                '[pymc :as pm]
                '[Bio.PDB.PDBParser]
                '[Bio.PDB]
                '[Bio.PDB.Polypeptide]
                '[pytensor]
                '[pytensor.tensor :as pt]
                '[math])

(defn brackets [obj entry]
  (py. obj __getitem__ entry))

(def colon
  (python/slice nil nil))

(arviz.style/use "arviz-darkgrid")


(defn extract-coordinates-from-pdb
  ([protein-name]
   (extract-coordinates-from-pdb protein-name {}))
  ([protein-name {:keys [data-type limit]
                  :or {data-type :models}}]
   (let [filepath (str "data/" protein-name ".pdb")
         parser (Bio.PDB/PDBParser)
         structure (py. parser get_structure protein-name filepath)]
     (case data-type
       :models (-> structure
                   (->> (map
                         (fn [model]
                           (-> model
                               (->> (mapcat
                                     (fn [chain]
                                       (->> chain
                                            (filter (fn [residue]
                                                      (-> residue
                                                          (py. get_resname)
                                                          (Bio.PDB.Polypeptide/is_aa :standard true))))
                                            (map (fn [residue]
                                                   (-> residue
                                                       (brackets "CA")
                                                       (py. get_coord)
                                                       (->> (dtype/->array :float32))))))))
                                    ((if limit
                                       (partial take limit)
                                       identity)))))))
                   (tensor/->tensor :datatype :float32))))))

(comment
  (-> "1d3z"
      extract-coordinates-from-pdb)

  (-> "1d3z"
      (extract-coordinates-from-pdb {:limit 4})))

(defn center-1d [xs]
  (fun/- xs
         (fun/mean xs)))

(defn center-columns [xyzs]
  (-> xyzs
      (tensor/map-axis center-1d 0)))

(comment
  (-> [[1 2 3]
       [4 5 9]]
      tensor/->tensor
      center-columns))

(defn center-columns [xyzs]
  (-> xyzs
      (tensor/map-axis center-1d 0)))


(defn xyz-tensor->dataset [tensor]
  (-> tensor
      dataset.tensor/tensor->dataset
      (tc/rename-columns [:x :y :z])
      #_(tc/add-column :i (fn [ds]
                            (-> ds tc/row-count range)))))

(defn read-data [prots
                 {:keys [data-type
                         models
                         rmsd?
                         limit]
                  :or {data-type :models
                       models [0 1]
                       rmsd true}}]
  (case data-type
    :models (let [coords (map (fn [prot model]
                                (-> prot
                                    (extract-coordinates-from-pdb {:limit limit})
                                    (nth model)))
                              prots
                              models)
                  obs (->> coords
                           (mapv #(tensor/map-axis % center-1d 0)))
                  obs-datasets (->> obs
                                    (mapv xyz-tensor->dataset))]
              {:coords coords
               :obs obs
               :obs-datasets obs-datasets})))


(let [name1 "1d3z"
      name2 "1ubq"
      models [0 0]
      samples 100]
  (->> (read-data [name1 name2]
                  {:models models})
       :obs-datasets
       (map tc/info)))



(defn compare-visually [xyz-datasets]
  (kind/hiccup
   ['(fn [{:keys [datasets index]}]
       [plotly
        {:data (->> datasets
                    (mapv (fn [dataset]
                            (->> dataset
                                 (merge {:type :scatter3d
                                         :mode :lines+markers
                                         :opacity 0.6
                                         :line {:width 10
                                                ;; :color "grey"
                                                }
                                         :marker {:size 4
                                                  :color index
                                                  :colorscale :Viridis}})))))}])
    {:datasets (->> xyz-datasets
                    (mapv #(update-vals % vec)))
     :index (-> xyz-datasets
                first
                tc/row-count
                range
                vec)}]))

(let [name1 "1d3z"
      name2 "1ubq"
      models [0 0]
      {:keys [obs-datasets]} (read-data [name1 name2]
                                        {:models models
                                         :limit 13})]
  (->> obs-datasets
       #_(take 1)
       compare-visually))


(defn ->max-distance-to-origin [centered-structure]
  (-> centered-structure
      fun/sq
      (tensor/reduce-axis fun/sum 1)
      fun/sqrt
      fun/reduce-max))

(defn ->average-structure [centered-structures]
  (-> centered-structures
      (->> (apply fun/+))
      (fun// (count centered-structures))))


;; trying PyTensor
;; https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html
(let [x (pt/scalar :name "x")
      y (pt/scalar :name "y")
      z (operator/add x y)
      w (pt/mul z 2)
      f (pytensor/function :inputs [x y]
                           :outputs w)]
  (f :x 10
     :y 5))



(defn ->quaternion [u]
  (let [theta1 (-> u
                   (brackets 1)
                   (operator/mul (* 2 Math/PI)))
        theta2 (-> u
                   (brackets 2)
                   (operator/mul (* 2 Math/PI)))
        r1 (-> u
               (brackets 0)
               (->> (operator/sub 1))
               pt/sqrt)
        r2 (-> u
               (brackets 0)
               pt/sqrt)
        w (-> theta2
              (pt/cos)
              (operator/mul r2))
        x (-> theta1
              (pt/sin)
              (operator/mul r1))
        y (-> theta1
              (pt/cos)
              (operator/mul r1))
        z (-> theta2
              (pt/sin)
              (operator/mul r2))
        R00 (operator/sub (operator/add (pt/sqr w)
                                        (pt/sqr x))
                          (operator/add (pt/sqr y)
                                        (pt/sqr z)))
        R11 (operator/sub (operator/add (pt/sqr w)
                                        (pt/sqr y))
                          (operator/add (pt/sqr x)
                                        (pt/sqr z)))
        R22 (operator/sub (operator/add (pt/sqr w)
                                        (pt/sqr z))
                          (operator/add (pt/sqr x)
                                        (pt/sqr y)))
        R01 (operator/mul 2
                          (operator/sub (operator/mul x y)
                                        (operator/mul w z)))
        R02 (operator/mul 2
                          (operator/add (operator/mul x z)
                                        (operator/mul w y)))
        R10 (operator/mul 2
                          (operator/add (operator/mul x y)
                                        (operator/mul w z)))
        R12 (operator/mul 2
                          (operator/sub (operator/mul y z)
                                        (operator/mul w x)))
        R20 (operator/mul 2
                          (operator/sub (operator/mul x z)
                                        (operator/mul w y)))
        R21 (operator/mul 2
                          (operator/add (operator/mul y z)
                                        (operator/mul w x)))]
    (pt/stack [(pt/stack [R00 R01 R02])
               (pt/stack [R10 R11 R12])
               (pt/stack [R20 R21 R22])])))

(py/with [model (pm/Model)]
         (let [x (pm/MatrixNormal "x"
                                  :mu (np/matrix [[0 1]
                                                  [3 4]])
                                  :colcov (np/matrix [[1 0]
                                                      [0 4]])
                                  :rowcov (np/matrix [[1 0]
                                                      [0 4]]))
               samples (pm/sample_prior_predictive)]
           samples))

(defn tensor2d->np-matrix [tensor]
  (->> tensor
       (map np/array)
       python/list
       np/matrix))


(-> [[1 2]
     [3 4]]
    tensor/->tensor
    tensor2d->np-matrix)





(def results
  (let [name1 "1d3z"
        name2 "1ubq"
        models [0 0]
        samples 100
        {:keys [obs obs-datasets]}
        (read-data [name1 name2]
                   {:models models
                    :limit 5})
        structures (->> obs
                        (mapv #(tensor/transpose % [1 0])))
        max-distance (->max-distance-to-origin (obs 0))
        average-structure (->average-structure obs)
        shape (-> (obs 0)
                  dtype/shape
                  reverse
                  vec)
        [space-dimension n-residues] shape]
    (py/with [model (pm/Model)]
             (let [M (pm/Normal "M" :shape shape)
                   M0 (pm/Deterministic "M0"
                                        (operator/sub
                                         M
                                         (pt/mean M)))
                   t (pm/Normal "t" :shape [space-dimension])
                   u (pm/Uniform "u" :shape [space-dimension])
                   R (pm/Deterministic "R" (->quaternion u))
                   U (pm/HalfNormal "U"
                                    :sigma 1
                                    :shape [n-residues])
                   M0_rotated (pm/Deterministic "M0_rotated"
                                                (pt/dot R M0))
                   dunmy (pm/Deterministic "dummy"
                                           (-> M0_rotated
                                               ;; conjugating with transpose
                                               ;; to make broadcasting work
                                               pt/transpose
                                               (operator/add t)
                                               pt/transpose))
                   X1 (pm/MatrixNormal "X1"
                                       :mu M0
                                       :rowcov (np/eye space-dimension)
                                       :colcov (pt/diag U)
                                       :observed (-> (structures 0)
                                                     tensor2d->np-matrix))
                   X2 (pm/MatrixNormal "X2"
                                       :mu
                                       (-> M0_rotated
                                           ;; conjugating with transpose
                                           ;; to make broadcasting work
                                           pt/transpose
                                           (operator/add t)
                                           pt/transpose)
                                       :rowcov (np/eye space-dimension)
                                       :colcov (pt/diag U)
                                       :observed (-> (structures 1)
                                                     tensor2d->np-matrix))
                   prior-predictive-samples (pm/sample_prior_predictive)
                   samples (pm/sample :chains 2
                                      :draws 100
                                      :tune 100)
                   posterior-predictive-samples (pm/sample_posterior_predictive
                                                 samples)]
               {:prior-predictive-samples prior-predictive-samples
                :posterior-predictive-samples posterior-predictive-samples
                :samples samples}))))



(-> results
    :prior-predictive-samples
    (py.- prior)
    (py.- "M0")
    np/mean) ; should be about zero

(defn py-array->clj [py-array]
  (let [np-array (np/array py-array)]
    (-> np-array
        (py. flatten)
        dtype/->double-array
        (tensor/reshape (np/shape np-array)))))

(-> results
    :samples
    (py.- posterior)
    (py.- u)
    py-array->clj
    first
    xyz-tensor->dataset
    ((fn [dataset]
       (kind/hiccup
        ['(fn [{:keys [dataset index]}]
            [plotly
             {:data [(->> dataset
                          (merge {:type :scatter3d
                                  :mode :markers
                                  :opacity 0.6
                                  :marker {:size 4
                                           :color index
                                           :colorscale :Viridis}}))]}])
         {:dataset (-> dataset
                       (update-vals vec))
          :index (-> dataset
                     tc/row-count
                     range
                     vec)}]))))



(let [posterior-predictives (-> results
                                :posterior-predictive-samples
                                (py.- posterior_predictive)
                                ((juxt #(py.- % X1)
                                       #(py.- % X2))))
      [n-chains n-samples _ _] (-> posterior-predictives
                                   first
                                   (py.- shape))]
  (for [chain-idx (range n-chains)
        sample-idx (range 0 n-samples (quot n-samples 5))]
    (->> posterior-predictives
         (map (fn [xarray]
                (-> xarray
                    (py. __getitem__ chain-idx)
                    (py. __getitem__ sample-idx)
                    np/array
                    py-array->clj
                    (tensor/transpose [1 0])
                    xyz-tensor->dataset)))
         compare-visually)))




(let [posteriors (-> results
                     :samples
                     (py.- posterior)
                     ((juxt #(py.- % M0)
                            #(py.- % M0_rotated))))
      [n-chains n-samples _ _] (-> posteriors
                                   first
                                   (py.- shape))]
  (for [chain-idx (range n-chains)
        sample-idx (range 0 n-samples (quot n-samples 5))]
    (->> posteriors
         (map (fn [xarray]
                (-> xarray
                    (py. __getitem__ chain-idx)
                    (py. __getitem__ sample-idx)
                    np/array
                    py-array->clj
                    (tensor/transpose [1 0])
                    xyz-tensor->dataset)))
         compare-visually)))
