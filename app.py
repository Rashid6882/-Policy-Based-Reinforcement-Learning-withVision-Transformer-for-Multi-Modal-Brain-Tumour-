import os
import json
import numpy as np
from flask import Flask, render_template, jsonify, send_file
import h5py

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    checkpoints_dir = 'checkpoints'
    
    dqn_stats = {}
    ddqn_stats = {}
    eval_summary = {}
    trajectories = []

    if os.path.exists(os.path.join(checkpoints_dir, 'dqn_history.json')):
        with open(os.path.join(checkpoints_dir, 'dqn_history.json'), 'r') as f:
            dqn_stats = json.load(f)

    if os.path.exists(os.path.join(checkpoints_dir, 'double_dqn_history.json')):
        with open(os.path.join(checkpoints_dir, 'double_dqn_history.json'), 'r') as f:
            ddqn_stats = json.load(f)

    if os.path.exists(os.path.join(checkpoints_dir, 'evaluation_summary.json')):
        with open(os.path.join(checkpoints_dir, 'evaluation_summary.json'), 'r') as f:
            eval_summary = json.load(f)

    if os.path.exists(os.path.join(checkpoints_dir, 'trajectories.json')):
        with open(os.path.join(checkpoints_dir, 'trajectories.json'), 'r') as f:
            trajectories = json.load(f)

    return jsonify({
        'dqn_history': dqn_stats,
        'double_dqn_history': ddqn_stats,
        'evaluation': eval_summary,
        'trajectories': trajectories
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
