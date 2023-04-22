(->> [prot-name1 prot-name2]
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
                                                                     (catch Exception e nil))})))))))))))))


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

(-> [protein-name1 protein-name2]
    (read-data {:limit 50})
    :obs-datasets
    compare-visually)
;; (defn ->max-distance-to-origin [centered-structure]
;;   (-> centered-structure
;;       fun/sq
;;       (tensor/reduce-axis fun/sum 1)
;;       fun/sqrt
;;       fun/reduce-max))

;; (defn ->average-structure [centered-structures]
;;   (-> centered-structures
;;       (->> (apply fun/+))
;;       (fun// (count centered-structures))))


;; trying PyTensor
;; https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html
#_(let [x (pt/scalar :name "x")
        y (pt/scalar :name "y")
        z (operator/add x y)
        w (pt/mul z 2)
        f (pytensor/function :inputs [x y]
                             :outputs w)]
    (f :x 10
       :y 5))



#_(py/with [model (pm/Model)]
           (let [x (pm/MatrixNormal "x"
                                    :mu (np/matrix [[0 1]
                                                    [3 4]])
                                    :colcov (np/matrix [[1 0]
                                                        [0 4]])
                                    :rowcov (np/matrix [[1 0]
                                                        [0 4]]))
                 samples (pm/sample_prior_predictive)]
             samples))

(let [view-limit 12
      tensor->cljs (fn [tensor]
                     (-> tensor
                         (tensor/transpose [1 0])
                         util/xyz-tensor->dataset
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
                      util/xyz-tensor->dataset
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
                                 util/xyz-tensor->dataset
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
                     util/xyz-tensor->dataset
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
