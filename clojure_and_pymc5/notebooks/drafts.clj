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
