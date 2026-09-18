import { useState } from "react";
import {
  Home,
  Upload,
  BarChart3,
  Info,
  Settings,
  CloudUpload,
  CheckCircle,
  Users,
  Target,
  Zap,
  Database,
  Trophy,
  Image as ImageIcon,
  Loader2,
} from "lucide-react";

import "./App.css";


// ============================================================
// SUPPORTED FRUIT CLASSES
// ============================================================

const fruits = [
  { name: "Apple", emoji: "🍎" },
  { name: "Banana", emoji: "🍌" },
  { name: "Orange", emoji: "🍊" },
  { name: "Pear", emoji: "🍐" },
  { name: "Strawberry", emoji: "🍓" },
  { name: "Pineapple", emoji: "🍍" },
  { name: "Watermelon", emoji: "🍉" },
];


// ============================================================
// MODEL ACCURACY
// These are your CNN model evaluation results.
// Update these numbers if your final Colab results differ.
// ============================================================

const accuracy = [
  { name: "SVM", value: 99.92 },
  { name: "KNN", value: 99.92 },
  { name: "Decision Tree", value: 97.19 },
  { name: "Random Forest", value: 100.00 },
];


function App() {

  // ==========================================================
  // STATE
  // ==========================================================

  const [selectedImage, setSelectedImage] = useState(null);

  const [activeMenu, setActiveMenu] = useState("Home");

  const [prediction, setPrediction] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState(null);


  // ==========================================================
  // IMAGE UPLOAD + PREDICTION
  // ==========================================================

  const handleImage = async (event) => {

    const file = event.target.files[0];

    if (!file) return;


    // --------------------------------------------------------
    // Validate image
    // --------------------------------------------------------

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      alert("Maximum file size is 5MB.");
      return;
    }


    // --------------------------------------------------------
    // Display selected image
    // --------------------------------------------------------

    const imageURL = URL.createObjectURL(file);

    setSelectedImage(imageURL);

    setPrediction(null);

    setError(null);

    setLoading(true);


    // --------------------------------------------------------
    // Send image to Flask
    // --------------------------------------------------------

    const formData = new FormData();

    formData.append("image", file);


    try {

      const response = await fetch(
        "https://potential-goldfish-jrqrjwwwqwg7c5vxx-5000.app.github.dev/predict",
        {
          method: "POST",
          body: formData,
        }
      );


      const data = await response.json();


      if (!response.ok) {

        throw new Error(
          data.message || "Prediction failed"
        );

      }


      // ------------------------------------------------------
      // Store prediction returned by Flask
      // ------------------------------------------------------

      setPrediction(data);

    } catch (err) {

      console.error("Prediction error:", err);

      setError(
        err.message ||
        "Unable to connect to the prediction server."
      );

    } finally {

      setLoading(false);

    }

  };


  // ==========================================================
  // HELPER FUNCTION
  // ==========================================================

  const getShortFruitName = (name) => {

    if (!name) return "-";

    return name
      .replace(" 10", "")
      .replace(" 1", "");

  };


  // ==========================================================
  // MODEL RESULTS
  // ==========================================================

  const modelResults = prediction
    ? [
        {
          model: "SVM",
          prediction: getShortFruitName(prediction.svm),
          confidence: "—",
          color: "purple",
        },
        {
          model: "KNN",
          prediction: getShortFruitName(prediction.knn),
          confidence: "—",
          color: "green",
        },
        {
          model: "Decision Tree",
          prediction: getShortFruitName(
            prediction.decision_tree
          ),
          confidence: "—",
          color: "orange",
        },
        {
          model: "Random Forest",
          prediction: getShortFruitName(
            prediction.random_forest
          ),
          confidence: "—",
          color: "red",
        },
      ]
    : [];


  // ==========================================================
  // FINAL PREDICTION
  // ==========================================================

  const finalPrediction = prediction
    ? getShortFruitName(
        prediction.final_prediction
      )
    : null;


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app">


      {/* ====================================================
          HEADER
      ==================================================== */}

      <header className="header">

        <div className="brand">

          <div className="brand-logo">
            🍎
          </div>

          <div>

            <h1>
              Fruit Classification
            </h1>

            <p>
              AI-Powered Fruit Recognition using CNN + Multiple Classifiers
            </p>

          </div>

        </div>


        <div className="header-right">

          <div className="model-badge">

            <span></span>

            7-Class Model

          </div>


          <div className="home-header">

            <Home size={18} />

            Home

          </div>

        </div>

      </header>


      <div className="layout">


        {/* ==================================================
            SIDEBAR
        ================================================== */}

        <aside className="sidebar">

          {[
            {
              name: "Home",
              icon: <Home size={22} />,
            },
            {
              name: "Upload & Predict",
              icon: <Upload size={22} />,
            },
            {
              name: "Model Performance",
              icon: <BarChart3 size={22} />,
            },
            {
              name: "About",
              icon: <Info size={22} />,
            },
            {
              name: "Settings",
              icon: <Settings size={22} />,
            },
          ].map((item) => (

            <button
              key={item.name}
              className={`side-item ${
                activeMenu === item.name
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setActiveMenu(item.name)
              }
            >

              {item.icon}

              <span>
                {item.name}
              </span>

            </button>

          ))}

        </aside>


        {/* ==================================================
            MAIN CONTENT
        ================================================== */}

        <main className="main">


          <div className="dashboard-grid">


            {/* =================================================
                LEFT COLUMN
            ================================================= */}

            <section className="left-column">


              {/* =================================================
                  WELCOME CARD
              ================================================= */}

              <div className="card welcome-card">

                <h2>
                  Welcome to Fruit Classification
                </h2>

                <p className="description">

                  Upload a fruit image and let our ensemble
                  model (MobileNetV2 + SVM, KNN, Decision Tree,
                  Random Forest) classify it using majority voting.

                </p>


                <div className="fruit-models">

                  {fruits.map((fruit) => (

                    <div
                      className="fruit-model"
                      key={fruit.name}
                    >

                      <div className="fruit-icon">

                        {fruit.emoji}

                      </div>

                      <span>
                        {fruit.name}
                      </span>

                    </div>

                  ))}

                </div>


                <div className="ensemble-banner">

                  <Users size={27} />

                  <span>
                    Ensemble Prediction
                  </span>

                  <strong>
                    →
                  </strong>

                  <span>
                    Majority Voting
                  </span>

                </div>

              </div>


              {/* =================================================
                  UPLOAD CARD
              ================================================= */}

              <div className="card upload-card">

                <h2>

                  <Upload size={23} />

                  Upload Fruit Image

                </h2>


                <label className="upload-box">


                  {selectedImage ? (

                    <div className="preview-container">

                      <img
                        src={selectedImage}
                        alt="Selected fruit"
                      />

                      <p>
                        Image selected successfully
                      </p>

                    </div>

                  ) : (

                    <>

                      <CloudUpload size={52} />

                      <h3>
                        Drag and drop an image here
                      </h3>

                      <span>
                        or
                      </span>

                      <div className="choose-button">

                        <ImageIcon size={18} />

                        Choose Image

                      </div>

                      <p>

                        Supported formats: JPG, PNG, JPEG

                        <span className="separator">
                          |
                        </span>

                        Max size: 5MB

                      </p>

                    </>

                  )}


                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    onChange={handleImage}
                    hidden
                  />

                </label>


                {/* =================================================
                    LOADING
                ================================================= */}

                {loading && (

                  <div className="loading-message">

                    <Loader2
                      size={22}
                      className="loading-spinner"
                    />

                    <span>
                      Analyzing image using MobileNetV2...
                    </span>

                  </div>

                )}


                {/* =================================================
                    ERROR
                ================================================= */}

                {error && (

                  <div className="error-message">

                    <strong>
                      Prediction failed
                    </strong>

                    <p>
                      {error}
                    </p>

                  </div>

                )}


                {/* =================================================
                    SAMPLE IMAGES
                ================================================= */}

                <div className="samples">

                  <h3>

                    <ImageIcon size={19} />

                    Sample Images

                  </h3>


                  <div className="sample-grid">

                    {fruits.slice(0, 6).map((fruit) => (

                      <div
                        className="sample"
                        key={fruit.name}
                      >

                        <div className="sample-image">

                          {fruit.emoji}

                        </div>

                        <span>
                          {fruit.name}
                        </span>

                      </div>

                    ))}

                  </div>

                </div>

              </div>

            </section>


            {/* =================================================
                RIGHT COLUMN
            ================================================= */}

            <section className="right-column">


              {/* =================================================
                  PREDICTION RESULT
              ================================================= */}

              <div className="card prediction-card">

                <div className="prediction-title">

                  <CheckCircle size={28} />

                  <h2>
                    Prediction Result
                  </h2>

                </div>


                <div className="prediction-main">


                  <div className="prediction-image">

                    {selectedImage ? (

                      <img
                        src={selectedImage}
                        alt="Fruit prediction"
                      />

                    ) : (

                      <span>
                        🍎
                      </span>

                    )}

                  </div>


                  <div className="prediction-info">

                    <span>
                      Predicted Fruit
                    </span>


                    <h1>

                      {finalPrediction || "—"}

                    </h1>


                    {prediction ? (

                      <>

                        <p>

                          Majority Vote:

                          {" "}

                          <strong>
                            {prediction.votes}/4
                          </strong>

                        </p>


                        <div className="confidence-bar">

                          <div
                            style={{
                              width: `${
                                (prediction.votes / 4) *
                                100
                              }%`,
                            }}
                          ></div>

                        </div>

                      </>

                    ) : (

                      <p>
                        Upload an image to get a prediction
                      </p>

                    )}

                  </div>

                </div>

              </div>


              {/* =================================================
                  MODEL PREDICTIONS
              ================================================= */}

              <div className="card model-card">

                <h2>
                  Model Predictions (Majority Voting)
                </h2>


                <div className="table-header">

                  <span>
                    Model
                  </span>

                  <span>
                    Prediction
                  </span>

                  <span>
                    Result
                  </span>

                </div>


                {prediction ? (

                  modelResults.map((result) => (

                    <div
                      className="model-row"
                      key={result.model}
                    >

                      <span className="model-name">

                        <i
                          className={`dot ${result.color}`}
                        ></i>

                        {result.model}

                      </span>


                      <strong className="prediction-green">

                        {result.prediction}

                      </strong>


                      <span>
                        ✓
                      </span>

                    </div>

                  ))

                ) : (

                  <div className="no-prediction">

                    Upload an image to see model predictions.

                  </div>

                )}


                {/* =================================================
                    FINAL RESULT
                ================================================= */}

                <div className="final-result">

                  <div className="trophy">

                    <Trophy size={24} />

                  </div>


                  <div>

                    <small>
                      Final Prediction (Majority Vote)
                    </small>

                    <h2>

                      {finalPrediction || "—"}

                    </h2>

                  </div>


                  <strong>

                    {prediction
                      ? `${prediction.votes} votes`
                      : "—"}

                  </strong>

                </div>

              </div>


              {/* =================================================
                  ACCURACY
              ================================================= */}

              <div className="card accuracy-card">

                <h2>
                  Model Accuracy Comparison
                </h2>


                <div className="chart">

                  {accuracy.map((item) => (

                    <div
                      className="bar-column"
                      key={item.name}
                    >

                      <span className="bar-value">

                        {item.value.toFixed(1)}%

                      </span>


                      <div className="bar-container">

                        <div
                          className="bar"
                          style={{
                            height: `${item.value}%`,
                          }}
                        ></div>

                      </div>


                      <span className="bar-name">

                        {item.name}

                      </span>

                    </div>

                  ))}

                </div>

              </div>

            </section>

          </div>


          {/* ====================================================
              FEATURE STRIP
          ==================================================== */}

          <div className="feature-strip">


            <div className="feature">

              <div className="feature-icon">

                <Target />

              </div>

              <div>

                <h3>
                  High Accuracy
                </h3>

                <p>
                  Ensemble learning improves classification performance
                </p>

              </div>

            </div>


            <div className="feature">

              <div className="feature-icon">

                <Zap />

              </div>

              <div>

                <h3>
                  Multiple Classifiers
                </h3>

                <p>
                  SVM, KNN, Decision Tree, Random Forest
                </p>

              </div>

            </div>


            <div className="feature">

              <div className="feature-icon">

                <Users />

              </div>

              <div>

                <h3>
                  Majority Voting
                </h3>

                <p>
                  Combines predictions for better reliability
                </p>

              </div>

            </div>


            <div className="feature">

              <div className="feature-icon">

                <Database />

              </div>

              <div>

                <h3>
                  Fruits-360 Dataset
                </h3>

                <p>
                  Fruit images used for model training
                </p>

              </div>

            </div>

          </div>


          {/* ====================================================
              FOOTER
          ==================================================== */}

          <footer>

            Fruit Classification Project

            <span>|</span>

            React Frontend

            <span>|</span>

            Flask Backend

            <span>|</span>

            CNN

            <span>|</span>

            4 Classifiers

            <span>|</span>

            Majority Voting

          </footer>


        </main>

      </div>

    </div>
  );
}


export default App;