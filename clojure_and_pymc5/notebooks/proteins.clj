(ns proteins
  (:require [tablecloth.api :as tc]
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
            [libpython-clj2.require :refer [require-python]]
            [util])
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

;; 1d3z
;; 1ubq
(def main-name1 "7ju5clean")
(def main-name2 "AF-A0A024R7T2-F1-model_v4-clean")

(->> [main-name1 main-name2]
     (mapv (fn [protein-name ]
             (let [filepath (str "data/" protein-name ".pdb")
                   parser (Bio.PDB/PDBParser)
                   structure (py. parser get_structure protein-name filepath)]
               (->> structure
                    first
                    ((fn [model]
                       (->> model
                            (mapcat (fn [chain]
                                      (->> chain
                                           (map (fn [residue]
                                                  (-> residue
                                                      (py. get_resname)))))))))))))))



(defn ->residue-type [residue]
  (-> residue
      str
      (subs 9 12)))

(def residues
  (->> [main-name1 main-name2]
       (mapv (fn [protein-name ]
               (let [filepath (str "data/" protein-name ".pdb")
                     parser (Bio.PDB/PDBParser)
                     structure (py. parser get_structure protein-name filepath)]
                 (->> structure
                      first
                      ((fn [model]
                         (->> model
                              (mapcat (fn [chain]
                                        (->> chain
                                             (map #(py. % get_resname))))))))))))))

(map count residues)


(-> (for [offset0 (range 20)
          offset1 (range 20)]
      {:offset0 offset0
       :offset1 offset1})
    (->> (map (fn [{:keys [offset0 offset1]
                    :as offsets}]
                (assoc offsets
                       :agreement (->> (map =
                                            (->> (residues 0)
                                                 (drop offset0))
                                            (->> (residues 1)
                                                 (drop offset1)))
                                       (filter identity)
                                       count)))))
    tc/dataset
    (tc/order-by [:agreement] :desc))


(defn extract-coordinates-from-pdb
  ([protein-name]
   (extract-coordinates-from-pdb protein-name {}))
  ([protein-name {:keys [data-type limit offset]
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
                                                   (try
                                                     (-> residue
                                                         (util/brackets "CA")
                                                         (py. get_coord)
                                                         (->> (dtype/->array :float32)))
                                                     (catch Exception e nil))))
                                            (filter some?))))
                                    ;; ((if (= protein-name main-name2)
                                    ;;    (comp rest reverse) ;; NOTE THIS !!!!
                                    ;;    identity))
                                    ((if offset
                                       (partial drop offset)
                                       identity))
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
                         limit
                         offsets]
                  :or {data-type :models
                       models [0 0]
                       offsets [0 0]
                       rmsd true}}]
  (case data-type
    :models (let [coords (map (fn [prot model offset]
                                (-> prot
                                    (extract-coordinates-from-pdb {:offset offset
                                                                   :limit limit})
                                    (nth model)))
                              prots
                              models
                              offsets)
                  obs (->> coords
                           (mapv #(tensor/map-axis % center-1d 0)))
                  obs-datasets (->> obs
                                    (mapv xyz-tensor->dataset))]
              {:coords coords
               :obs obs
               :obs-datasets obs-datasets})))



(let [name1 main-name1
      name2 main-name2
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
                                         :opacity 0.5
                                         :line {:width 5}
                                         :marker {:size 4
                                                  :color index
                                                  :colorscale :Viridis}})))))}])
    {:datasets (->> xyz-datasets
                    (mapv util/prep-dataset-for-cljs))
     :index (-> xyz-datasets
                first
                tc/row-count
                range
                vec)}]))

(let [name1 main-name1
      name2 main-name2
      models [0 0]
      offsets [10 0]
      {:keys [obs-datasets]} (read-data [name1 name2]
                                        {:models models
                                         :offsets offsets
                                         :limit 50})]
  (->> obs-datasets
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



(defn rotate-q [u]
  (let [theta1 (-> u
                   (util/brackets 1)
                   (operator/mul (* 2 Math/PI)))
        theta2 (-> u
                   (util/brackets 2)
                   (operator/mul (* 2 Math/PI)))
        r1 (-> u
               (util/brackets 0)
               (->> (operator/sub 1))
               pt/sqrt)
        r2 (-> u
               (util/brackets 0)
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





(let [name1 main-name1
      name2 main-name2
      models [0 0]
      offsets [10 0]
      {:keys [obs obs-datasets]}
      (read-data [name1 name2]
                 {:models models
                  :offsets offsets
                  ;; :limit 50
                  })
      structures (->> obs
                      (mapv #(-> %
                                 (tensor/transpose [1 0]))))
      view-limit 50
      tensor->cljs (fn [tensor]
                     (-> tensor
                         (tensor/transpose [1 0])
                         xyz-tensor->dataset
                         (tc/head view-limit)
                         util/prep-dataset-for-cljs))]
  (->> {:prot1-dataset  (-> structures
                            first
                            tensor->cljs)
        :prot2-dataset (-> structures
                           second
                           tensor->cljs)
        :index (-> structures
                   (->> (map dtype/shape))
                   pr-str)}
       (vector '(fn [{:keys [prot1-dataset
                             prot2-dataset
                             index]}]
                  [:div
                   index
                   [plotly
                    {:data [(-> prot1-dataset
                                (merge {:type :scatter3d
                                        :mode :lines+markers
                                        :opacity 1
                                        :marker {:size 3
                                                 :color "purple"}}))
                            (-> prot2-dataset
                                (merge {:type :scatter3d
                                        :mode :lines+markers
                                        :opacity 1
                                        :marker {:size 3
                                                 :color "orange"}}))]}]]))
       kind/hiccup))




(def results
  (let [name1 main-name1
        name2 main-name2
        models [0 0]
        offsets [10 0]
        samples 100
        {:keys [obs obs-datasets]}
        (read-data [name1 name2]
                   {:models models
                    :offsets offsets
                    :limit 100})
        structures (->> obs
                        (mapv #(-> %
                                   (tensor/transpose [1 0]))))
        np-structures (->> structures
                           (mapv tensor2d->np-matrix))
        ;; max-distance (->max-distance-to-origin (obs 0))
        ;; average-structure (->average-structure obs)
        shape (-> (obs 0)
                  dtype/shape
                  reverse
                  vec)
        [space-dimension n-residues] shape]
    (py/with [model (pm/Model)]
             (let [M (pm/Cauchy "M"
                                :alpha 0
                                :beta 1
                                :shape shape)
                   M0 (pm/Deterministic "M0"
                                        (operator/sub
                                         M
                                         (pt/mean M)))
                   t (pm/Normal "t" :shape [space-dimension]) ; the shift
                   u (pm/Uniform "u" :shape [space-dimension]) ; randomization of rotation
                   R (pm/Deterministic "R" (rotate-q u)) ; the rotation matrix
                   U (pm/HalfNormal "U"
                                    :sigma 0.01 ; TODO: Consider some prior here
                                    :shape [n-residues])
                   M0_rotated (pm/Deterministic "M0_rotated"
                                                (pt/dot R M0))
                   X1 (pm/MatrixNormal "X1"
                                       :mu M0
                                       :rowcov (np/eye space-dimension)
                                       :colcov (pt/diag U)
                                       :observed (np-structures 0))
                   X2 (pm/MatrixNormal "X2"
                                       :mu (-> M0_rotated
                                               ;; conjugating with transpose
                                               ;; to make broadcasting work
                                               pt/transpose
                                               (operator/add t)
                                               pt/transpose)
                                       :rowcov (np/eye space-dimension)
                                       :colcov (pt/diag U)
                                       :observed (np-structures 1))
                   M0_adapted (pm/Deterministic "M0_adapted"
                                                (-> (pt/dot R M0)
                                                    pt/transpose
                                                    (operator/add t)
                                                    pt/transpose))
                   X1_adapted (pm/Deterministic "X1_adapted"
                                                (-> (pt/dot R X1)
                                                    pt/transpose
                                                    (operator/add t)
                                                    pt/transpose))
                   prot1_adapted (pm/Deterministic "prot1_adapted"
                                                   (-> (np-structures 0)
                                                       (->> (pt/dot R))
                                                       pt/transpose
                                                       (operator/add t)
                                                       pt/transpose))
                   prior-predictive-samples (pm/sample_prior_predictive)
                   idata (pm/sample :chains 1
                                    :draws 500
                                    :tune 500)
                   posterior-predictive-samples (pm/sample_posterior_predictive
                                                 idata)]
               {:structures structures
                :prior-predictive-samples prior-predictive-samples
                :posterior-predictive-samples posterior-predictive-samples
                :idata idata}))))




(let [view-limit 100
      tensor->cljs (fn [tensor aname]
                     (-> tensor
                         (tensor/transpose [1 0])
                         xyz-tensor->dataset
                         (tc/head view-limit)
                         util/prep-dataset-for-cljs))
      shape (-> results
                :idata
                (py.- posterior)
                (py.- prot1_adapted)
                np/shape)
      n-chains (first shape)
      n-samples (second shape)]
  (->> {:prot1-adapted-datasets
        (-> results
            :idata
            (py.- posterior)
            (py.- prot1_adapted)
            util/py-array->clj
            (tensor/slice 1)
            (->> (map-indexed
                  (fn [chain-idx chain-tensor]
                    (-> chain-tensor
                        (tensor/slice 1)
                        (->> (map #(tensor->cljs
                                    %
                                    (str "prot1-adapted-chain"
                                         chain-idx)))))))
                 (apply concat)
                 vec))
        :prot1-chain-idx (->> n-chains
                              range
                              (mapcat (fn [chain-idx]
                                        (repeat n-samples chain-idx)))
                              vec)
        :prot2-dataset
        (-> results
            :structures
            second
            (tensor->cljs "prot2"))}
       (vector '(fn [{:keys [prot1-adapted-datasets
                             prot1-chain-idx
                             prot2-dataset]}]
                  [plotly
                   {:data (->> prot1-adapted-datasets
                               (map (fn [dataset]
                                      (-> dataset
                                          (merge {:type :scatter3d
                                                  :mode :lines+markers
                                                  :opacity 0.05
                                                  :marker {:size 3
                                                           :color
                                                           (mapv
                                                            ["blue"
                                                             "yellow"
                                                             "red"
                                                             "green"]
                                                            prot1-chain-idx)}}))))
                               (cons (-> prot2-dataset
                                         (merge {:type :scatter3d
                                                 :mode :lines+markers
                                                 :opacity 1
                                                 :marker {:size 3
                                                          :color "orange"}})))
                               vec)}]))
       kind/hiccup))








(let [view-limit 12
      tensor->cljs (fn [tensor]
                     (-> tensor
                         (tensor/transpose [1 0])
                         xyz-tensor->dataset
                         (tc/head view-limit)
                         util/prep-dataset-for-cljs))]
  (->> {:prot1-adapted-datasets (-> results
                                    :idata
                                    (py.- posterior)
                                    (py.- prot1_adapted)
                                    util/py-array->clj
                                    (tensor/slice 2)
                                    (->> ;; (partition 19)
                                     ;; (map first)
                                     (mapv tensor->cljs)))
        :prot2-dataset (-> results
                           :structures
                           second
                           tensor->cljs)}
       (vector '(fn [{:keys [prot1-adapted-datasets
                             prot2-dataset]}]
                  [plotly
                   {:data (->> prot1-adapted-datasets
                               (map (fn [dataset]
                                      (-> dataset
                                          (merge {:type :scatter3d
                                                  :mode :lines+markers
                                                  :opacity 0.01
                                                  :marker {:size 3
                                                           :color "grey"}}))))
                               (cons (-> prot2-dataset
                                         (merge {:type :scatter3d
                                                 :mode :lines+markers
                                                 :opacity 1
                                                 :marker {:size 3
                                                          :color "orange"}})))
                               vec)}]))
       kind/hiccup))






(let [posterior-predictives (-> results
                                :posterior-predictive-samples
                                (py.- posterior_predictive)
                                ((juxt #(py.- % X1)
                                       #(py.- % X1_adapted)
                                       #(py.- % X2))))
      view-limit 6
      [n-chains n-samples _ _] (-> posterior-predictives
                                   first
                                   (py.- shape))]
  (for [chain-idx [0] #_(range n-chains)
        sample-idx (range 0 n-samples (quot n-samples 10))]
    (kind/hiccup
     [:div
      (pr-str {:chain chain-idx
               :sample sample-idx})
      (->> posterior-predictives
           (map (fn [xarray]
                  (-> xarray
                      (py. __getitem__ chain-idx)
                      (py. __getitem__ sample-idx)
                      np/array
                      util/py-array->clj
                      (tensor/transpose [1 0])
                      xyz-tensor->dataset
                      (tc/head view-limit))))
           compare-visually)])))


(kind/hiccup
 ['(fn [{:keys [datasets]}]
     [plotly
      {:data (->> datasets
                  (mapv (fn [dataset]
                          (-> dataset
                              (merge {:type :scatter3d
                                      :mode :lines+markers
                                      :opacity 0.3
                                      :marker {:size 3
                                               :color "grey"}})))))}])
  {:datasets (-> results
                 :idata
                 (py.- posterior)
                 (py.- M0_adapted)
                 util/py-array->clj
                 (tensor/transpose [0 1 3 2])
                 (tensor/slice 2)
                 (->> (map (fn [tensor]
                             (-> tensor
                                 xyz-tensor->dataset
                                 util/prep-dataset-for-cljs)))
                      vec))}])


(-> results
    :idata
    (py.- posterior)
    (py.- prot1_adapted)
    util/py-array->clj)


(-> results
    :idata
    (py.- posterior)
    (py.- u)
    (->> (mapv (fn [chain-posterior]
                 (-> chain-posterior
                     util/py-array->clj
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
                          (let [index (-> dataset
                                          tc/row-count
                                          range
                                          vec)]
                            {:dataset (-> dataset
                                          (tc/add-column
                                           :i index)
                                          util/prep-dataset-for-cljs)
                             :index index})]))))))))



(-> results
    :prior-predictive-samples
    (py.- prior)
    (py.- "M0")
    np/mean) ; should be about zero





;; some better prior for U?
;; add a RMSE measure
;; color the chains differently
;; visualize varying variance:
;;   by the tube's thickness
;;   by color (blue/grey/red)
;; contour plots -- need to understand
;; use b-factor from the pdb file
;; b/(2*PI) -> rms deviation



:bye
