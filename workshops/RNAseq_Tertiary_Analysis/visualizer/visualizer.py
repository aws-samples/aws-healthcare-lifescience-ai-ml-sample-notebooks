import os
from pyvis.network import Network
from sagemaker.lineage.artifact import Artifact

class Visualizer:
    def __init__(self):
        self.directory = "generated"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def render(self, query_lineage_response, scenario_name, sagemaker_session):
        net = self.get_network()
        for vertex in query_lineage_response["Vertices"]:
            arn = vertex["Arn"]
            if "Type" in vertex:
                label = vertex["Type"]
            else:
                label = None
            lineage_type = vertex["LineageType"]
            name = self.get_name(arn, label, lineage_type, sagemaker_session)
            title = self.get_title(arn, label, lineage_type)
            color = self.get_color(lineage_type)
            net.add_node(
                vertex["Arn"],
                label=name,
                title=title,
                shape="box",
                physics="false",
                color=color,
            )

        for edge in query_lineage_response["Edges"]:
            source = edge["SourceArn"]
            dest = edge["DestinationArn"]
            net.add_edge(source, dest)

        return net.show(f"{self.directory}/{scenario_name}.html")

    def get_title(self, arn, label, lineage_type):
        return f"Arn: {arn} Type: {label} Lineage Type: {lineage_type}"

    def get_name(self, arn, label, lineage_type, sagemaker_session):
        if lineage_type == "Artifact":
            return (
                label
                + " "
                + Artifact.load(
                    artifact_arn=arn,
                    sagemaker_session=sagemaker_session,
                ).source.source_uri
            )
        else:
            name = arn.split("/")[1]
            return label + " " + name

    def get_network(self):
        net = Network(height="800px", width="100%", directed=True, notebook=True)
        return net

    def get_color(self, lineage_type):
        if lineage_type == "Context":
            return "yellow"
        elif lineage_type == "Artifact":
            return "orange"
        else:
            return None
        