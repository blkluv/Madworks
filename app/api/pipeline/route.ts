import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    const maybeImage = formData.get('image');
    // Validate presence of a file in the 'image' field to avoid Python 422s
    if (!(maybeImage instanceof File)) {
      return NextResponse.json(
        {
          error: "Missing required file field 'image' in multipart/form-data.",
          hint: "Attach an image using the 'image' field, or call /api/pipeline/run for the full pipeline which handles optional images.",
        },
        { status: 400 }
      );
    }

    // Get the pipeline URL from environment variables with fallback to localhost
    const pipelineUrl = process.env.NEXT_PUBLIC_PIPELINE_URL || 'http://localhost:8010';
    
    // Forward only the verified image to the Python service
    const forward = new FormData();
    forward.append('image', maybeImage, (maybeImage as File).name || 'upload.png');
    const response = await fetch(`${pipelineUrl}/ingest-analyze`, {
      method: 'POST',
      body: forward,
    });

    // Get the response data
    const data = await response.json();

    // Return the response from the Python service
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Error in pipeline route:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Add CORS headers
export function OPTIONS() {
  return new NextResponse(null, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
