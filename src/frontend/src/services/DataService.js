import { BASE_API_URL } from "./Common";

const axios = require('axios');

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    GetExperiments: async function () {
        return await axios.get(BASE_API_URL + "/experiments");
    },
    GetCurrentmodel: async function () {
        return await axios.get(BASE_API_URL + "/best_model");
    },
    Predict: async function (formData) {
        return await axios.post(BASE_API_URL + "/predict", formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    },
}

export default DataService;