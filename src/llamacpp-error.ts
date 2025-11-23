import { createJsonErrorResponseHandler } from "@ai-sdk/provider-utils";
import { z } from "zod/v4";

const llamacppErrorDataSchema = z.object({
  error: z.object({
    code: z.number().or(z.string()).optional(),
    message: z.string(),
    type: z.string().optional(),
  }),
});

export type LlamacppErrorData = z.infer<typeof llamacppErrorDataSchema>;

export const llamacppFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: llamacppErrorDataSchema,
  errorToMessage: (data) => data.error.message,
});
