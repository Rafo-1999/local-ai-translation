package reactiveQuarkus;

import io.smallrye.mutiny.Uni;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;


@Produces(MediaType.APPLICATION_JSON)
public class SupesResource {

    @GET
    public String hello(){return "hello";}

    @GET
    @Path("/greeting")
    public Uni<String>greeting(){
        return Uni.createFrom().item("greeting");
    }
}
