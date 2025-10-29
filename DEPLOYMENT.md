# Deployment to Render

This guide will help you deploy your Chest X-Ray Classification app to Render.

## Prerequisites

1. A GitHub account
2. A Render account (sign up at [render.com](https://render.com))
3. Your model file at `outputs_multi/model_multiclass.pth`

## Step-by-Step Deployment

### Step 1: Push to GitHub

1. Initialize a git repository if not already done:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. Create a new repository on GitHub

3. Push your code:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Render

#### Option A: Using Render Dashboard (Recommended for beginners)

1. Go to [Render Dashboard](https://dashboard.render.com)

2. Click **"New +"** → **"Web Service"**

3. Connect your GitHub repository

4. Configure the service:
   - **Name**: `chest-xray-classifier` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `python -m pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python -m gunicorn app_multiclass_dynamic:app --bind 0.0.0.0:$PORT`
   - **Plan**: Choose a plan (Free tier available for testing)

5. Add Environment Variables:
   - Key: `MODEL_PATH`, Value: `outputs_multi/model_multiclass.pth`

6. Click **"Create Web Service"**

#### Option B: Using Render Blueprint (render.yaml)

1. In Render Dashboard, click **"New +"** → **"Blueprint"**

2. Connect your GitHub repository that contains `render.yaml`

3. Render will automatically detect and configure the service

4. Click **"Apply"**

### Step 3: Important Notes

⚠️ **Model File Size**
- Your model file (`outputs_multi/model_multiclass.pth`) may be large
- Render's free tier has a 500MB disk space limit
- If your model exceeds this, consider:
  - Using Render Disk to mount external storage
  - Compressing the model file
  - Using a paid tier with more storage

### Step 4: Test Your Deployment

Once deployed, you'll get a URL like: `https://your-app-name.onrender.com`

1. Visit the URL in your browser
2. Upload a chest X-ray image
3. Check if predictions work correctly

## Environment Variables

You can add these optional environment variables in Render dashboard:

- `MODEL_PATH`: Path to your model file (default: `outputs_multi/model_multiclass.pth`)

## Troubleshooting Questions

### Build fails with "No module named 'torch'"
- Check that `requirements.txt` exists and has all dependencies
- Ensure file paths are correct in the repository

### Model file not found
- Make sure `outputs_multi/model_multiclass.pth` is committed to GitHub
- Check file size limits
- Verify the file path in environment variables

### Upload functionality not working
- Check that `static/uploads/` directory exists
- Ensure file permissions are correct
- Check Render logs for errors

## Monitoring

- **Logs**: View in Render Dashboard → Your Service → Logs
- **Metrics**: CPU, Memory, and Disk usage
- **Events**: Deployment history and events

## Performance Tips

1. **Use Disk Storage**: For large model files, upgrade to a plan with persistent disk
2. **Optimize Model**: Consider quantizing or using ONNX for smaller model size
3. **Caching**: Consider adding Redis for session management
4. **CDN**: For static assets, consider using a CDN

## Cost Considerations

- **Free Tier**: 
  - 750 hours/month
  - 500MB disk space
  - Spins down after 15 minutes of inactivity
- **Starter Plan**: $7/month
  - Always on
  - 512MB RAM
  - 1GB disk space

## Support

If you encounter issues:
1. Check Render documentation: https://render.com/docs
2. View application logs in Render Dashboard
3. Check the Render Status page: https://status.render.com

---

**Good luck with your deployment! 🚀**
