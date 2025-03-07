(ns proteins
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
            [util])
  (:import java.lang.Math))

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

(def protein-name1 "7ju5clean")
(def protein-name2 "AF-A0A024R7T2-F1-model_v4-clean")
(def protein-names [protein-name1 protein-name2])

(defn name->path [nam]
  (str "data/" nam ".pdb"))

(defonce protein-name->pdb-data
  (memoize
   (fn [protein-name]
     (-> protein-name
         name->path
         slurp))))

(defn pdb-view [pdb]
  (kind/hiccup
   ['(fn [pdb]
       [:div
        {:style {:width "100%"
                 :height "500px"
                 :position "relative"}
         :ref (fn [el]
                (let [config (clj->js
                              {:backgroundColor "0xffffff"})
                      viewer (.createViewer js/$3Dmol el #_config)]
                  (.addModelsAsFrames viewer pdb "pdb")
                  (.setStyle viewer
                             (clj->js {})
                             (clj->js {:stick {:color :spectrum}}))
                  (.addSphere viewer (clj->js
                                      {:center {:x 0
                                                :y 0
                                                :z 0}
                                       :radius 1
                                       :color "green"}))
                  (.zoomTo viewer)
                  (.render viewer)
                  (.zoom viewer 0.8 1000)))}
        ;; need to keep this symbol to let Clay infer the necessary dependency
        'three-d-mol])
    pdb]))

(->> protein-names
     (map (fn [nam]
            [nam
             (-> nam
                 protein-name->pdb-data
                 pdb-view)]) )
     (into {}))


(defn shapes-view [shapes]
  (kind/hiccup
   ['(fn [shapes]
       [:div
        {:style {:width "100%"
                 :height "500px"
                 :position "relative"}
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
    (vec shapes)]))

(-> [[:sphere {:center {:x 0
                        :y 0
                        :z 0}
               :radius 1
               :color "green"}]
     [:cylinder {:start {:x 0 :y 10 :z 20}
                 :end {:x 10 :y 0 :z 30}
                 :radius 0.5
                 :fromCap false
                 :toCap true
                 :color :teal
                 :alpha 0.5}]]
    shapes-view)


(defn extract-coordinates-from-pdb
  ([protein-name]
   (let [filepath (name->path protein-name)
         parser (Bio.PDB/PDBParser)
         structure (py. parser get_structure protein-name filepath)]
     (-> structure
         first
         ((fn [model]
            (-> model
                (->> (mapcat
                      (fn [chain]
                        (->> chain
                             (filter (fn [residue]
                                       (-> residue
                                           (py. get_resname)
                                           (Bio.PDB.Polypeptide/is_aa :standard true))))
                             (map (fn [residue]
                                    {:id (-> residue
                                             (py. get_id)
                                             second)
                                     :name (-> residue
                                               (py. get_resname))
                                     :ca-coordinates (try
                                                       (-> residue
                                                           (util/brackets "CA")
                                                           (py. get_coord)
                                                           (->> (dtype/->array :float32)))
                                                       (catch Exception e nil))}))
                             (filter :ca-coordinates))))
                     tc/dataset))))))))


(-> protein-name1
    extract-coordinates-from-pdb
    ;; for readability of output:
    (tc/update-columns [:ca-coordinates]
                       (partial map vec)))

(defn center-1d [xs]
  (fun/- xs
         (fun/mean xs)))

(defn center-columns [xyzs]
  (-> xyzs
      (tensor/map-axis center-1d 0)))

(defn read-data
  ([prots]
   (read-data prots nil))
  ([prots {:keys [limit]}]
   (let [prots [protein-name1 protein-name2]
         [dataset1 dataset2] (->> prots
                                  (map extract-coordinates-from-pdb))
         joined-dataset (-> (tc/inner-join dataset1 dataset2 :id)
                            ((if limit
                               #(tc/head % limit)
                               identity)))
         coords (->> [:ca-coordinates :right.ca-coordinates]
                     (map (fn [colname]
                            (-> colname
                                joined-dataset
                                tensor/->tensor))))
         obs (->> coords
                  (mapv #(tensor/map-axis % center-1d 0)))
         obs-datasets (->> obs
                           (mapv util/xyz-tensor->dataset))]
     {:coords coords
      :obs obs
      :obs-datasets obs-datasets})))


(-> [protein-name1 protein-name2]
    (read-data {:limit 4})
    :obs-datasets)


;; Compare the datasets visually

(defn xyz-dataset->shapes [dataset options]
  (-> dataset
      (tc/rows :as-maps)
      (->> ((juxt identity rest))
           (apply mapv (fn [xyz0 xyz1]
                         [:cylinder (merge {:start xyz0
                                            :end xyz1}
                                           options)])))))


(let [{:keys [obs obs-datasets]} (-> [protein-name1 protein-name2]
                                     read-data)
      colors [:purple :orange]
      radii [1 1]
      view-limit 50]
  (->> [obs-datasets colors radii]
       (apply mapcat
              (fn [dataset color radius]
                (-> dataset
                    (tc/head view-limit)
                    (xyz-dataset->shapes {:radius radius
                                          :color color}))))
       shapes-view))


(let [{:keys [obs obs-datasets]} (-> [protein-name1 protein-name2]
                                     read-data)
      structures (->> obs
                      (mapv #(-> %
                                 (tensor/transpose [1 0]))))
      view-limit 50
      tensor->cljs (fn [tensor]
                     (-> tensor
                         (tensor/transpose [1 0])
                         util/xyz-tensor->dataset
                         (tc/head view-limit)
                         util/prep-dataset-for-cljs))]
  (->> {:prot1-dataset  (-> structures
                            first
                            tensor->cljs)
        :prot2-dataset (-> structures
                           second
                           tensor->cljs)}
       (vector '(fn [{:keys [prot1-dataset
                             prot2-dataset]}]
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
                                                :color "orange"}}))]}]))
       kind/hiccup))


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


(defonce model
  (memoize
   (fn [{:keys [residues-limit tune]}]
     (let [{:keys [obs obs-datasets]}
           (read-data [protein-name1 protein-name2]
                      {:limit residues-limit})
           structures (->> obs
                           (mapv #(-> %
                                      (tensor/transpose [1 0]))))
           np-structures (->> structures
                              (mapv util/tensor2d->np-matrix))
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
                                       :draws 200
                                       :tune tune)
                      posterior-predictive-samples (pm/sample_posterior_predictive
                                                    idata)]
                  {:structures structures
                   :prior-predictive-samples prior-predictive-samples
                   :posterior-predictive-samples posterior-predictive-samples
                   :idata idata}))))))


(defn view-results-3dmol [results {:keys [residues-view-limit
                                          samples-view-limit]}]
  (let [tensor->dataset (fn [tensor]
                          (-> tensor
                              (tensor/transpose [1 0])
                              util/xyz-tensor->dataset
                              (tc/head residues-view-limit)))
        shape (-> results
                  :idata
                  (py.- posterior)
                  (py.- prot1_adapted)
                  np/shape)
        n-chains (first shape)
        n-samples (second shape)
        shape (-> results
                  :idata
                  (py.- posterior)
                  (py.- prot1_adapted)
                  np/shape)
        n-chains (first shape)
        n-samples (second shape)
        prot1-adapted-tensors (-> results
                                  :idata
                                  (py.- posterior)
                                  (py.- prot1_adapted)
                                  util/py-array->clj
                                  (tensor/slice 1)
                                  (->> (map #(tensor/slice % 1))
                                       (apply concat)))
        prot1-adapted-datasets (->> prot1-adapted-tensors
                                    (mapv tensor->dataset))
        prot2-tensor (-> results
                         :structures
                         second)
        prot2-dataset (-> prot2-tensor
                          tensor->dataset)
        mean-rmsd (->> prot1-adapted-tensors
                       (map (fn [prot1-tensor]
                              (-> (fun/- prot1-tensor
                                         prot2-tensor)
                                  fun/sq
                                  fun/mean
                                  fun/sqrt)))
                       fun/mean)
        prot1-chain-idx (->> n-chains
                             range
                             (mapcat (fn [chain-idx]
                                       (repeat n-samples chain-idx)))
                             vec)]
    [{:mean-rmsd mean-rmsd}
     (->> prot1-adapted-datasets
          (take samples-view-limit)
          (mapcat #(xyz-dataset->shapes
                    %
                    {:alpha 0.6
                     :radius 0.3
                     :color :purple}))
          (concat (-> prot2-dataset
                      (xyz-dataset->shapes
                       {:radius 0.3
                        :color :orange})))
          shapes-view)]))



(defn view-results [results {:keys [residues-view-limit
                                    samples-view-limit]}]
  (let [tensor->cljs (fn [tensor aname]
                       (-> tensor
                           (tensor/transpose [1 0])
                           util/xyz-tensor->dataset
                           (tc/head residues-view-limit)
                           util/prep-dataset-for-cljs))
        shape (-> results
                  :idata
                  (py.- posterior)
                  (py.- prot1_adapted)
                  np/shape)
        n-chains (first shape)
        n-samples (second shape)
        prot1-adapted-tensors (-> results
                                  :idata
                                  (py.- posterior)
                                  (py.- prot1_adapted)
                                  util/py-array->clj
                                  (tensor/slice 1)
                                  (->> (map #(tensor/slice % 1))
                                       (apply concat)))
        prot1-adapted-datasets (->> prot1-adapted-tensors
                                    (map-indexed (fn [i tensor]
                                                   (-> tensor
                                                       (tensor->cljs (str "prot1-" i)))))
                                    vec)
        prot2-tensor (-> results
                         :structures
                         second)
        prot2-dataset (-> prot2-tensor
                          (tensor->cljs "prot2"))
        mean-rmsd (->> prot1-adapted-tensors
                       (map (fn [prot1-tensor]
                              (-> (fun/- prot1-tensor
                                         prot2-tensor)
                                  fun/sq
                                  fun/mean
                                  fun/sqrt)))
                       fun/mean)
        prot1-chain-idx (->> n-chains
                             range
                             (mapcat (fn [chain-idx]
                                       (repeat n-samples chain-idx)))
                             vec)]
    (->> {:prot1-adapted-datasets prot1-adapted-datasets
          :prot1-chain-idx prot1-chain-idx
          :prot2-dataset prot2-dataset}
         (vector '(fn [{:keys [prot1-adapted-datasets
                               prot1-chain-idx
                               prot2-dataset]}]
                    [plotly
                     {:data (->> prot1-adapted-datasets
                                 (map (fn [dataset]
                                        (-> dataset
                                            (merge {:type :scatter3d
                                                    :mode :lines+markers
                                                    :opacity 0.1
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
         kind/hiccup)))


(defn report [{:keys [tune
                      residues-limit]}]
  (let [m (model {:residues-limit residues-limit
                  :tune tune})]
    #_{:tune tune}
    #_(view-results m {:residues-view-limit 100
                       :samples-view-limit 10})
    (view-results-3dmol m {:residues-view-limit 50
                           :samples-view-limit 10})))


(comment
  (->> [5 15 50 200]
       (mapcat (fn [tune]
                 (report {:tune tune}))))

  (-> {:tune 5
       :residues-limit 50}
      model
      (view-results-3dmol {:residues-view-limit 100
                           :samples-view-limit 10})))





:bye
