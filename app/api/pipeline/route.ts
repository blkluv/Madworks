import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();
    
    // Get the pipeline URL from environment variables with fallback to localhost
    const pipelineUrl = process.env.NEXT_PUBLIC_PIPELINE_URL || 'http://localhost:8010';
    
    // Forward the request to the Python service
    const response = await fetch(`${pipelineUrl}/ingest-analyze`, {
      method: 'POST',
      body: formData,
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
